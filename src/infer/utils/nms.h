#pragma once

#include "cvkit/core/types.h"

#include <vector>

namespace cvkit::infer::detail
{

    struct Candidate
    {
        cvkit::core::BBox box{};
        float             score{0.0F};
        int               class_id{-1};
    };

    [[nodiscard]] std::vector<cvkit::core::Detection> non_maximum_suppression(
        std::vector<Candidate>          candidates,
        const std::vector<std::string>& labels,
        float                           iou_threshold);

}  // namespace cvkit::infer::detail
