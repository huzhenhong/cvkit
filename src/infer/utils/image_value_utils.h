#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/task_io.h"

#include <optional>
#include <string>

namespace cvkit::infer::detail
{

    [[nodiscard]] std::optional<cvkit::core::Frame> materialize_host_frame(
        const cvkit::infer::ImageValue& image,
        std::string*                    error_message = nullptr);

}  // namespace cvkit::infer::detail
