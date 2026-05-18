#pragma once

#include "cvkit/infer/model.h"

#include "../backends/backend_session.h"

#include <memory>
#include <string_view>
#include <vector>

namespace cvkit::infer::detail
{

    struct PipelineContext
    {
        ModelSpec                spec{};
        std::vector<std::string> labels{};
        float                    confidence_threshold{0.25F};
        float                    iou_threshold{0.45F};
    };

    class ITaskPipeline
    {
      public:
        virtual ~ITaskPipeline() = default;

        virtual TaskKind   task() const   = 0;
        virtual TaskSchema schema() const = 0;
        virtual TaskOutput run_sync(
            const IBackendSession& backend,
            const TaskInput&       input,
            const PipelineContext& context) const = 0;
    };

    std::unique_ptr<ITaskPipeline> create_task_pipeline(TaskKind task, std::string_view family);

}  // namespace cvkit::infer::detail
