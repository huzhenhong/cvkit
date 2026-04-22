#pragma once

#include "cvkit/core/types.h"

#include <cstdint>
#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] bool                      is_nchw_layout(const std::vector<std::int64_t>& shape);

    [[nodiscard]] std::vector<std::int64_t> resolve_input_shape(
        const std::vector<std::int64_t>& input_shape,
        const cvkit::core::Frame&        frame);

}  // namespace cvkit::infer::detail
