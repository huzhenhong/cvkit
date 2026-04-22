#pragma once

#include "../task_pipeline.h"

namespace cvkit::infer::detail
{

    class PromptableSegmentationPipeline final : public ITaskPipeline
    {
      public:
        [[nodiscard]] TaskKind   task() const override;
        [[nodiscard]] TaskSchema schema() const override;
        [[nodiscard]] TaskOutput run_sync(
            const IBackendSession& backend,
            const TaskInput&       input,
            const PipelineContext& context) const override;
    };

}  // namespace cvkit::infer::detail
