#pragma once

#include "cvkit/infer/infer_export.h"

#include <cstdint>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT Backend : std::uint8_t
    {
        none,
        onnxruntime,
        tensorrt,
    };

}  // namespace cvkit::infer
