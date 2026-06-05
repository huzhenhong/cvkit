#pragma once

#include "../backends/backend_session.h"

#include <optional>
#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] std::optional<cvkit::core::Frame> resolve_host_image_input(const TaskInput& input);

    [[nodiscard]] std::vector<std::int64_t> resolve_nchw_input_shape(
        const IBackendSession&    backend,
        const cvkit::core::Frame& frame);

    [[nodiscard]] RawTensor build_rgb_nchw_float_input(
        const cvkit::core::Frame&        frame,
        const std::vector<std::int64_t>& input_shape,
        std::string                      tensor_name = "images");

}  // namespace cvkit::infer::detail
