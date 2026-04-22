#pragma once

#include "cvkit/infer/infer_export.h"

#include <cstdint>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT DeviceKind : std::uint8_t
    {
        cpu,
        cuda,
        npu,
    };

    enum class BK_INFER_EXPORT MemoryDevice : std::uint8_t
    {
        host,
        cuda,
        npu,
    };

    struct BK_INFER_EXPORT DeviceRef
    {
        DeviceKind kind{DeviceKind::cpu};
        int        index{0};
    };

}  // namespace cvkit::infer
