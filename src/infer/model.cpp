#include "cvkit/infer/model.h"

#include "executor.h"
#include "backends/backend_session.h"
#include "graph/graph.h"
#include "tasks/task_pipeline.h"
#include "tile_runner.h"
#include "utils/labels.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::infer
{

    class Model::Impl
    {
      public:
        struct TraceState
        {
            std::mutex                  mutex{};
            std::vector<GraphTraceInfo> last_trace{};
        };

        ModelSpec                                spec_{};
        std::shared_ptr<detail::IBackendSession> backend_{};
        std::shared_ptr<detail::ITaskPipeline>   pipeline_{};
        std::shared_ptr<detail::IExecutor>       executor_{detail::create_default_executor()};
        std::shared_ptr<TraceState>              trace_state_{std::make_shared<TraceState>()};
        std::vector<std::string>                 labels_{};
        float                                    confidence_threshold_{0.25F};
        float                                    iou_threshold_{0.45F};
        TileOptions                              tile_options_{};
        bool                                     loaded_{false};
    };

    namespace
    {

        [[nodiscard]] Backend normalize_backend(Backend backend)
        {
            if (backend == Backend::none)
            {
#if defined(CVKIT_WITH_ONNXRUNTIME)
                return Backend::onnxruntime;
#elif defined(CVKIT_WITH_TENSORRT)
                return Backend::tensorrt;
#else
                return Backend::none;
#endif
            }
            return backend;
        }

        [[nodiscard]] TaskKind normalize_task(TaskKind task)
        {
            if (task == TaskKind::unknown)
            {
                return TaskKind::detection;
            }
            return task;
        }

        [[nodiscard]] std::string normalize_family(TaskKind task, std::string family)
        {
            if (!family.empty())
            {
                return family;
            }

            switch (task)
            {
                case TaskKind::detection:
                    return "yolo11";
                case TaskKind::face_detection:
                    return "scrfd";
                case TaskKind::classification:
                    return "classification";
                case TaskKind::segmentation:
                    return "segmentation";
                case TaskKind::pose:
                    return "pose";
                case TaskKind::facemesh:
                    return "facemesh";
                case TaskKind::promptable_segmentation:
                    return "sam";
                case TaskKind::unknown:
                default:
                    return {};
            }
        }

        [[nodiscard]] TaskFuture make_ready_future(TaskOutput output)
        {
            std::promise<TaskOutput> promise;
            auto                     future = promise.get_future().share();
            promise.set_value(std::move(output));
            return TaskFuture{std::move(future)};
        }

        template<typename ImplT>
        [[nodiscard]] detail::PipelineContext make_pipeline_context(const ImplT& impl)
        {
            return detail::PipelineContext{
                impl.spec_,
                impl.labels_,
                impl.confidence_threshold_,
                impl.iou_threshold_};
        }

        template<typename ImplT>
        [[nodiscard]] detail::TaskGraph make_pipeline_graph(const ImplT& impl)
        {
            return detail::create_pipeline_graph(impl.backend_, impl.pipeline_, make_pipeline_context(impl));
        }

        template<typename ImplT>
        [[nodiscard]] bool has_runtime_state(const ImplT& impl)
        {
            return impl.loaded_ && impl.backend_ != nullptr && impl.pipeline_ != nullptr;
        }

        template<typename SessionT>
        void append_session_tensors(std::vector<TensorInfo>& destination, const SessionT& session, bool inputs)
        {
            for (std::size_t index = 0;; ++index)
            {
                const auto* tensor = inputs ? session.input_info(index) : session.output_info(index);
                if (tensor == nullptr)
                {
                    break;
                }
                destination.push_back(*tensor);
            }
        }

        [[nodiscard]] GraphInfo convert_graph_info(const detail::TaskGraph& graph)
        {
            GraphInfo info{};
            for (const auto& metadata : graph.metadata())
            {
                info.nodes.push_back(GraphNodeInfo{
                    metadata.name,
                    metadata.depends_on,
                    metadata.consumes,
                    metadata.produces});
            }

            const auto boundary   = graph.boundary();
            info.boundary.inputs  = boundary.inputs;
            info.boundary.outputs = boundary.outputs;
            return info;
        }

        [[nodiscard]] std::vector<GraphTraceInfo> convert_graph_trace(const std::vector<detail::NodeTrace>& trace)
        {
            std::vector<GraphTraceInfo> result;
            result.reserve(trace.size());
            for (const auto& node : trace)
            {
                result.push_back(GraphTraceInfo{
                    node.name,
                    node.sequence,
                    node.input_count,
                    node.output_count,
                    node.scratch_count,
                    node.duration_us,
                    node.ok,
                    node.message});
            }
            return result;
        }

        template<typename TraceStateT>
        void store_last_trace(const std::shared_ptr<TraceStateT>& trace_state, const detail::Packet& packet)
        {
            std::lock_guard<std::mutex> lock(trace_state->mutex);
            trace_state->last_trace = convert_graph_trace(packet.trace);
        }

        template<typename TraceStateT>
        void clear_last_trace(const std::shared_ptr<TraceStateT>& trace_state)
        {
            std::lock_guard<std::mutex> lock(trace_state->mutex);
            trace_state->last_trace.clear();
        }

    }  // namespace

    Model::Model()
        : impl_(std::make_unique<Impl>())
    {
    }

    Model::~Model() = default;

    Model::Model(Model&&) noexcept            = default;
    Model& Model::operator=(Model&&) noexcept = default;

    bool   Model::load(const ModelSpec& spec)
    {
        impl_->loaded_ = false;
        impl_->backend_.reset();
        impl_->pipeline_.reset();
        impl_->labels_.clear();

        auto normalized    = spec;
        normalized.backend = normalize_backend(spec.backend);
        normalized.task    = normalize_task(spec.task);
        normalized.family  = normalize_family(normalized.task, spec.family);
        impl_->spec_       = normalized;

        auto backend = detail::create_backend_session(normalized.backend);
        if (backend == nullptr || !backend->load(normalized) || !backend->ready())
        {
            return false;
        }

        auto pipeline = detail::create_task_pipeline(normalized.task, normalized.family);
        if (pipeline == nullptr)
        {
            return false;
        }

        impl_->backend_  = std::shared_ptr<detail::IBackendSession>(std::move(backend));
        impl_->pipeline_ = std::shared_ptr<detail::ITaskPipeline>(std::move(pipeline));
        impl_->loaded_   = true;

        if (!normalized.labels_path.empty())
        {
            return load_labels(normalized.labels_path);
        }

        return true;
    }

    bool Model::load_labels(std::string labels_path)
    {
        std::vector<std::string> labels;
        if (!detail::load_labels_file(labels_path, labels))
        {
            return false;
        }

        impl_->spec_.labels_path = std::move(labels_path);
        impl_->labels_           = std::move(labels);
        return true;
    }

    bool Model::loaded() const
    {
        return impl_->loaded_;
    }

    Backend Model::backend() const
    {
        return impl_->backend_ != nullptr ? impl_->backend_->backend() : impl_->spec_.backend;
    }

    TaskKind Model::task() const
    {
        return impl_->pipeline_ != nullptr ? impl_->pipeline_->task() : impl_->spec_.task;
    }

    TaskSchema Model::schema() const
    {
        return impl_->pipeline_ != nullptr ? impl_->pipeline_->schema() : TaskSchema{};
    }

    SessionInfo Model::session_info() const
    {
        SessionInfo info{};
        if (impl_->backend_ == nullptr)
        {
            return info;
        }

        append_session_tensors(info.inputs, *impl_->backend_, true);
        append_session_tensors(info.outputs, *impl_->backend_, false);

        return info;
    }

    GraphInfo Model::graph_info() const
    {
        if (!has_runtime_state(*impl_))
        {
            return {};
        }

        return convert_graph_info(make_pipeline_graph(*impl_));
    }

    std::vector<GraphTraceInfo> Model::last_graph_trace() const
    {
        std::lock_guard<std::mutex> lock(impl_->trace_state_->mutex);
        return impl_->trace_state_->last_trace;
    }

    std::string_view Model::model_path() const
    {
        return impl_->spec_.model_path;
    }

    std::string_view Model::aux_model_path() const
    {
        return impl_->spec_.aux_model_path;
    }

    std::string_view Model::labels_path() const
    {
        return impl_->spec_.labels_path;
    }

    std::string_view Model::family() const
    {
        return impl_->spec_.family;
    }

    std::string_view Model::cache_dir() const
    {
        return impl_->spec_.cache_dir;
    }

    CachePolicy Model::cache_policy() const
    {
        return impl_->spec_.cache_policy;
    }

    void Model::set_confidence_threshold(float threshold)
    {
        impl_->confidence_threshold_ = threshold;
    }

    float Model::confidence_threshold() const
    {
        return impl_->confidence_threshold_;
    }

    void Model::set_iou_threshold(float threshold)
    {
        impl_->iou_threshold_ = threshold;
    }

    float Model::iou_threshold() const
    {
        return impl_->iou_threshold_;
    }

    void Model::set_tile_options(const TileOptions& options)
    {
        impl_->tile_options_ = options;
    }

    TileOptions Model::tile_options() const
    {
        return impl_->tile_options_;
    }

    TaskOutput Model::run_sync(const TaskInput& input) const
    {
        if (!has_runtime_state(*impl_))
        {
            return {};
        }

        const auto* tile_source = detail::find_tiled_source_frame(input);
        if (detail::should_run_tiled(impl_->spec_.task, impl_->tile_options_, tile_source))
        {
            return detail::run_tiled_sync(
                impl_->backend_,
                impl_->pipeline_,
                make_pipeline_context(*impl_),
                input,
                impl_->tile_options_,
                *tile_source);
        }

        auto           graph = make_pipeline_graph(*impl_);
        detail::Packet packet{};
        packet.input = input;
        auto result  = graph.run_sync(std::move(packet));
        store_last_trace(impl_->trace_state_, result);
        return std::move(result.output);
    }

    TaskFuture Model::submit(const TaskInput& input) const
    {
        if (!has_runtime_state(*impl_))
        {
            return make_ready_future({});
        }

        auto           backend      = impl_->backend_;
        auto           pipeline     = impl_->pipeline_;
        auto           copied_input = input;
        auto           context      = make_pipeline_context(*impl_);
        auto           trace_state  = impl_->trace_state_;
        auto           tile_options = impl_->tile_options_;
        const auto*    tile_source  = detail::find_tiled_source_frame(input);
        if (detail::should_run_tiled(context.spec.task, tile_options, tile_source))
        {
            auto source_frame = std::make_shared<cvkit::core::Frame>(*tile_source);
            auto shared_input = std::make_shared<TaskInput>(std::move(copied_input));
            auto shared_context = std::make_shared<detail::PipelineContext>(std::move(context));
            clear_last_trace(trace_state);
            return impl_->executor_->submit(
                [
                    backend = std::move(backend),
                    pipeline = std::move(pipeline),
                    shared_input = std::move(shared_input),
                    shared_context = std::move(shared_context),
                    tile_options,
                    source_frame = std::move(source_frame)
                ]()
                {
                    return detail::run_tiled_sync(
                        backend,
                        pipeline,
                        *shared_context,
                        *shared_input,
                        tile_options,
                        *source_frame);
                });
        }

        auto           graph        = detail::create_pipeline_graph(std::move(backend), std::move(pipeline), std::move(context));
        detail::Packet packet{};
        packet.input = std::move(copied_input);
        clear_last_trace(trace_state);
        return impl_->executor_->submit([graph = std::move(graph), packet = std::move(packet), trace_state = std::move(trace_state)]() mutable
                                        {
            auto result = graph.submit_packet(std::move(packet)).get();
            store_last_trace(trace_state, result);
            return std::move(result.output); });
    }

    std::vector<cvkit::core::Detection> Model::run_detection(const cvkit::core::Frame& frame) const
    {
        TaskInput input{};
        input.add("image", frame);

        auto output = run_sync(input);
        if (const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
            detections != nullptr)
        {
            return *detections;
        }

        return {};
    }

}  // namespace cvkit::infer
