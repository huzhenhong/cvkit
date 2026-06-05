#pragma once

#include "cvkit/infer/model.h"

#include "backends/backend_session.h"
#include "tasks/task_pipeline.h"

#include <memory>

namespace cvkit::infer::detail
{

    const cvkit::core::Frame* find_tiled_source_frame(const TaskInput& input);

    bool should_run_tiled(TaskKind task, const TileOptions& options, const cvkit::core::Frame* frame);

    TaskOutput run_tiled_sync(
        const std::shared_ptr<IBackendSession>& backend,
        const std::shared_ptr<ITaskPipeline>&   pipeline,
        const PipelineContext&                  context,
        const TaskInput&                        input,
        const TileOptions&                      options,
        const cvkit::core::Frame&               source_frame);

}  // namespace cvkit::infer::detail
