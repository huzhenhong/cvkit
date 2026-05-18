#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/device.h"
#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/session.h"
#include "cvkit/infer/task.h"
#include "cvkit/infer/task_io.h"
#include "cvkit/infer/task_schema.h"

#include "cvkit/core/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT CachePolicy : std::uint8_t
    {
        default_policy,
        disabled,
        read_only,
        rebuild,
    };

    struct BK_INFER_EXPORT TensorRtProfileShape
    {
        std::vector<std::int64_t> min{};
        std::vector<std::int64_t> opt{};
        std::vector<std::int64_t> max{};
    };

    struct BK_INFER_EXPORT TensorRtProfileSpec
    {
        std::string          input_name{};
        TensorRtProfileShape shape{};
    };

    struct BK_INFER_EXPORT ModelSpec
    {
        std::string                      model_path{};
        std::string                      aux_model_path{};
        std::string                      labels_path{};
        std::string                      cache_dir{};
        Backend                          backend{Backend::none};
        TaskKind                         task{TaskKind::unknown};
        CachePolicy                      cache_policy{CachePolicy::default_policy};
        std::string                      family{};
        DeviceRef                        device{};
        std::vector<TensorRtProfileSpec> tensorrt_profiles{};
        bool                             tensorrt_prefer_device_outputs{false};
    };

    struct BK_INFER_EXPORT GraphNodeInfo
    {
        std::string              name{};
        std::vector<std::string> depends_on{};
        std::vector<std::string> consumes{};
        std::vector<std::string> produces{};
    };

    struct BK_INFER_EXPORT GraphBoundaryInfo
    {
        std::vector<std::string> inputs{};
        std::vector<std::string> outputs{};
    };

    struct BK_INFER_EXPORT GraphInfo
    {
        std::vector<GraphNodeInfo> nodes{};
        GraphBoundaryInfo          boundary{};
    };

    struct BK_INFER_EXPORT GraphTraceInfo
    {
        std::string   name{};
        std::size_t   sequence{0};
        std::size_t   input_count{0};
        std::size_t   output_count{0};
        std::size_t   scratch_count{0};
        std::uint64_t duration_us{0};
        bool          ok{true};
        std::string   message{};
    };

    class BK_INFER_EXPORT Model
    {
      public:
        Model();
        ~Model();

        Model(Model&&) noexcept;
        Model& operator=(Model&&) noexcept;

        Model(const Model&)                                                       = delete;
        Model&                                            operator=(const Model&) = delete;

        bool                                              load(const ModelSpec& spec);
        bool                                              load_labels(std::string labels_path);
        bool                                              loaded() const;
        Backend                                           backend() const;
        TaskKind                                          task() const;
        TaskSchema                                        schema() const;
        SessionInfo                                       session_info() const;
        GraphInfo                                         graph_info() const;
        std::vector<GraphTraceInfo>                       last_graph_trace() const;
        std::string_view                                  model_path() const;
        std::string_view                                  aux_model_path() const;
        std::string_view                                  labels_path() const;
        std::string_view                                  family() const;
        std::string_view                                  cache_dir() const;
        CachePolicy                                       cache_policy() const;
        void                                              set_confidence_threshold(float threshold);
        float                                             confidence_threshold() const;
        void                                              set_iou_threshold(float threshold);
        float                                             iou_threshold() const;
        [[nodiscard]] TaskOutput                          run_sync(const TaskInput& input) const;
        [[nodiscard]] TaskFuture                          submit(const TaskInput& input) const;
        [[nodiscard]] std::vector<cvkit::core::Detection> run_detection(const cvkit::core::Frame& frame) const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::infer
