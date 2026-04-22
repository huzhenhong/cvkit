#pragma once

#include "cvkit/core/types.h"
#include "../../../backends/backend_session.h"
#include "yolo_preprocess_cpu.h"

#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] std::vector<cvkit::core::Detection> postprocess_yolo_detections(
        const RawTensorMap&               outputs,
        const LetterboxResult&            preprocess,
        const cvkit::core::Frame&         frame,
        const std::vector<std::string>&   labels,
        float                             confidence_threshold,
        float                             iou_threshold);

}  // namespace cvkit::infer::detail
