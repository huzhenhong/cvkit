#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/task_io.h"

#include <opencv2/core.hpp>

#include <string>

namespace cvkit::examples
{

    [[nodiscard]] cvkit::core::Frame       mat_to_frame(const cv::Mat& image, std::string source);
    [[nodiscard]] cvkit::infer::ImageValue mat_to_image_value(const cv::Mat& image, std::string source);
    [[nodiscard]] cv::Mat                  frame_mask_to_mat(const cvkit::core::Frame& mask);

}  // namespace cvkit::examples
