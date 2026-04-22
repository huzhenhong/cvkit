#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/task.h"
#include "cvkit/infer/task_io.h"

namespace cvkit::infer
{

    class BK_INFER_EXPORT Session
    {
      public:
        virtual ~Session() = default;

        [[nodiscard]] virtual bool       ready() const                = 0;
        [[nodiscard]] virtual Backend    backend() const              = 0;
        [[nodiscard]] virtual TaskKind   task() const                 = 0;
        [[nodiscard]] virtual TaskOutput run_sync(const TaskInput& input) const = 0;
        [[nodiscard]] virtual TaskFuture submit(const TaskInput& input) const   = 0;
    };

}  // namespace cvkit::infer
