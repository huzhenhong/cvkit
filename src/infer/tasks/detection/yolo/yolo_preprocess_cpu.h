#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/infer_export.h"
#include "../../../backends/backend_session.h"

namespace cvkit::infer::detail
{

    struct LetterboxResult
    {
        RawTensor tensor{};
        float     scale{1.0F};
        float     pad_x{0.0F};
        float     pad_y{0.0F};
        int       input_width{0};
        int       input_height{0};
    };

    [[nodiscard]] BK_INFER_EXPORT LetterboxResult preprocess_yolo_cpu(
        const cvkit::core::Frame&        frame,
        const std::vector<std::int64_t>& input_shape);

}  // namespace cvkit::infer::detail
