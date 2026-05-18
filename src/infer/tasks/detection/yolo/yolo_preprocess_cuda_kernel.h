#pragma once

#include "cvkit/infer/task_io.h"

#include "yolo_preprocess_cpu.h"

#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] bool preprocess_yolo_cuda_kernel(
        const cvkit::infer::ImageValue& image,
        const std::vector<std::int64_t>& input_shape,
        bool                            prefer_device_tensor_output,
        LetterboxResult&                 result);

}  // namespace cvkit::infer::detail
