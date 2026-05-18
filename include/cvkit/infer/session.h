#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/task.h"
#include "cvkit/infer/task_io.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cvkit::infer
{

    using TensorMap = std::vector<TensorValue>;

    struct BK_INFER_EXPORT TensorInfo
    {
        std::string               name{};
        std::vector<std::int64_t> shape{};
        TensorDataType            data_type{TensorDataType::float32};
        MemoryDevice              memory_device{MemoryDevice::host};

        [[nodiscard]] bool        is_host_accessible() const
        {
            return memory_device == MemoryDevice::host;
        }
    };

    struct BK_INFER_EXPORT SessionInfo
    {
        std::vector<TensorInfo> inputs{};
        std::vector<TensorInfo> outputs{};
    };

    class BK_INFER_EXPORT Session
    {
      public:
        virtual ~Session() = default;

        virtual bool       ready() const                          = 0;
        virtual Backend    backend() const                        = 0;
        virtual TaskKind   task() const                           = 0;
        virtual TaskOutput run_sync(const TaskInput& input) const = 0;
        virtual TaskFuture submit(const TaskInput& input) const   = 0;
    };

}  // namespace cvkit::infer
