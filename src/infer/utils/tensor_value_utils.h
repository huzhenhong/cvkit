#pragma once

#include "cvkit/infer/task_io.h"

#include <optional>
#include <string>

namespace cvkit::infer::detail
{

    [[nodiscard]] std::optional<cvkit::infer::TensorValue> materialize_host_tensor(
        const cvkit::infer::TensorValue& tensor,
        std::string*                     error_message = nullptr);

}  // namespace cvkit::infer::detail
