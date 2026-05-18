#pragma once

#include "cvkit/infer/infer_export.h"

#include "../task_pipeline.h"

namespace cvkit::infer::detail
{

    class BK_INFER_EXPORT ClassificationPipeline final : public ITaskPipeline
    {
      public:
        TaskKind   task() const override;
        TaskSchema schema() const override;
        TaskOutput run_sync(
            const IBackendSession& backend,
            const TaskInput&       input,
            const PipelineContext& context) const override;
    };

}  // namespace cvkit::infer::detail
