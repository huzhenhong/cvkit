#include "cvkit/infer/model.h"

#include "executor.h"
#include "backends/backend_session.h"
#include "tasks/task_pipeline.h"
#include "utils/labels.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::infer
{

    class Model::Impl
    {
      public:
        ModelSpec                               spec_{};
        std::shared_ptr<detail::IBackendSession> backend_{};
        std::shared_ptr<detail::ITaskPipeline>   pipeline_{};
        std::shared_ptr<detail::IExecutor>       executor_{detail::create_default_executor()};
        std::vector<std::string>                 labels_{};
        float                                    confidence_threshold_{0.25F};
        float                                    iou_threshold_{0.45F};
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

    bool Model::load(std::string model_path)
    {
        ModelSpec spec{};
        spec.model_path = std::move(model_path);
        spec.backend    = Backend::none;
        spec.task       = TaskKind::detection;
        spec.family     = "yolo11";
        return load(spec);
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

    std::string_view Model::model_path() const
    {
        return impl_->spec_.model_path;
    }

    std::string_view Model::labels_path() const
    {
        return impl_->spec_.labels_path;
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

    TaskOutput Model::run_sync(const TaskInput& input) const
    {
        if (!impl_->loaded_ || impl_->backend_ == nullptr || impl_->pipeline_ == nullptr)
        {
            return {};
        }

        const detail::PipelineContext context{
            impl_->labels_,
            impl_->confidence_threshold_,
            impl_->iou_threshold_};

        return impl_->pipeline_->run_sync(*impl_->backend_, input, context);
    }

    TaskFuture Model::submit(const TaskInput& input) const
    {
        if (!impl_->loaded_ || impl_->backend_ == nullptr || impl_->pipeline_ == nullptr)
        {
            return make_ready_future({});
        }

        auto backend = impl_->backend_;
        auto pipeline = impl_->pipeline_;
        auto copied_input = input;
        detail::PipelineContext context{
            impl_->labels_,
            impl_->confidence_threshold_,
            impl_->iou_threshold_};

        return impl_->executor_->submit(
            [backend = std::move(backend),
             pipeline = std::move(pipeline),
             input = std::move(copied_input),
             context = std::move(context)]() mutable
            {
                return pipeline->run_sync(*backend, input, context);
            });
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

    std::vector<cvkit::core::Detection> Model::run(const cvkit::core::Frame& frame) const
    {
        return run_detection(frame);
    }

}  // namespace cvkit::infer
