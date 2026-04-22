#include "promptable_segmentation_pipeline.h"

#include "cvkit/infer/tasks/promptable_segmentation.h"

namespace cvkit::infer::detail
{

    TaskKind PromptableSegmentationPipeline::task() const
    {
        return TaskKind::promptable_segmentation;
    }

    TaskSchema PromptableSegmentationPipeline::schema() const
    {
        return promptable_segmentation_schema();
    }

    TaskOutput PromptableSegmentationPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        static_cast<void>(backend);
        static_cast<void>(input);
        static_cast<void>(context);
        return {};
    }

}  // namespace cvkit::infer::detail
