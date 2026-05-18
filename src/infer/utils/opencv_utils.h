#pragma once

#include "cvkit/core/types.h"

#include <opencv2/core.hpp>

namespace cvkit::infer::detail
{

    [[nodiscard]] cv::Mat frame_to_mat_copy(const cvkit::core::Frame& frame);

}  // namespace cvkit::infer::detail
