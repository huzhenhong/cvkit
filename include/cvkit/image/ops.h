#pragma once

#include "cvkit/image/image_export.h"

#include "cvkit/core/types.h"

namespace cvkit::image
{

    [[nodiscard]] BK_IMAGE_EXPORT cvkit::core::Frame resize(
        const cvkit::core::Frame& input,
        int                       width,
        int                       height);

}  // namespace cvkit::image
