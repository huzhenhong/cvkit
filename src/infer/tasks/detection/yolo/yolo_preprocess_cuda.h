#pragma once

#include "cvkit/infer/task_io.h"
#include "cvkit/infer/infer_export.h"
#include "cvkit/core/types.h"

#include "yolo_preprocess_cpu.h"

#include <optional>
#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] BK_INFER_EXPORT std::optional<cvkit::core::Frame> copy_cuda_image_to_host_frame(
        const cvkit::infer::ImageValue& image,
        std::string*                    error_message = nullptr);

    [[nodiscard]] BK_INFER_EXPORT std::optional<LetterboxResult> preprocess_yolo_cuda(
        const cvkit::infer::ImageValue&  image,
        const std::vector<std::int64_t>& input_shape,
        bool                             prefer_device_tensor_output = false,
        std::string*                     error_message               = nullptr);

}  // namespace cvkit::infer::detail
