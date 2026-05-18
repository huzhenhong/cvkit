#pragma once

#include "../task_pipeline.h"
#include "../../graph/packet.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] bool prepare_detection_inference(
        const IBackendSession&   backend,
        const TaskInput&         input,
        Packet&                  packet);

    [[nodiscard]] bool prepare_detection_inference_async(
        const IBackendSession&   backend,
        const TaskInput&         input,
        Packet&                  packet,
        BackendFuture&           future);

    [[nodiscard]] std::vector<cvkit::core::Detection> finalize_detection_output(
        const Packet&            packet,
        const PipelineContext&   context);

    class DetectionPipeline final : public ITaskPipeline
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
