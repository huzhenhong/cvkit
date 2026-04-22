#include "bbox.h"

#include <algorithm>

namespace cvkit::infer::detail
{

    float intersection_over_union(const cvkit::core::BBox& lhs, const cvkit::core::BBox& rhs)
    {
        const auto left   = std::max(lhs.x, rhs.x);
        const auto top    = std::max(lhs.y, rhs.y);
        const auto right  = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
        const auto bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);

        const auto width        = std::max(0.0F, right - left);
        const auto height       = std::max(0.0F, bottom - top);
        const auto intersection = width * height;
        const auto union_area   = lhs.width * lhs.height + rhs.width * rhs.height - intersection;
        if (union_area <= 0.0F)
        {
            return 0.0F;
        }

        return intersection / union_area;
    }

}  // namespace cvkit::infer::detail
