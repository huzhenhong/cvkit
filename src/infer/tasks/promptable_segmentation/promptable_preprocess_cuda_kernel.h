#pragma once

#include "cvkit/infer/task_io.h"

#include "../../backends/backend_session.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] bool preprocess_promptable_encoder_cuda_kernel(
        const cvkit::infer::ImageValue& image,
        bool                            prefer_device_tensor_output,
        RawTensor&                      tensor);

}  // namespace cvkit::infer::detail
