#pragma once

#include "cvkit/infer/task_io.h"

#include "../../backends/backend_session.h"

#include <optional>
#include <string>

namespace cvkit::infer::detail
{

    [[nodiscard]] std::optional<RawTensor> preprocess_promptable_encoder_cuda(
        const cvkit::infer::ImageValue& image,
        bool                            prefer_device_tensor_output = false,
        std::string*                    error_message = nullptr);

}  // namespace cvkit::infer::detail
