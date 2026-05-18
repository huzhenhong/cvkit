#pragma once

#include "../task_pipeline.h"
#include "../../graph/packet.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] bool prepare_promptable_segmentation_inference(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context,
        Packet&                packet);

    [[nodiscard]] TaskOutput finalize_promptable_segmentation_output(
        const Packet&          packet,
        const PipelineContext& context);

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
