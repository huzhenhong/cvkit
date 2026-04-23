#pragma once

#include "cvkit/infer/task_io.h"

#include <functional>
#include <memory>

namespace cvkit::infer::detail
{

    class IExecutor
    {
      public:
        virtual ~IExecutor() = default;

        [[nodiscard]] virtual TaskFuture submit(std::function<TaskOutput()> task) = 0;
    };

    [[nodiscard]] std::shared_ptr<IExecutor> create_default_executor();

}  // namespace cvkit::infer::detail
