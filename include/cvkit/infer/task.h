#pragma once

#include "cvkit/infer/infer_export.h"

#include <cstdint>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT TaskKind : std::uint8_t
    {
        unknown,
        detection,
        classification,
        promptable_segmentation,
    };

}  // namespace cvkit::infer
