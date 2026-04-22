#pragma once

#include "cvkit/core/types.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] float intersection_over_union(const cvkit::core::BBox& lhs, const cvkit::core::BBox& rhs);

}  // namespace cvkit::infer::detail
