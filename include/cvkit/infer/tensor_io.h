#pragma once

#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/task_io.h"

#include <filesystem>

namespace cvkit::infer
{

    [[nodiscard]] BK_INFER_EXPORT bool save_tensor_file(
        const TensorValue&           tensor,
        const std::filesystem::path& path);

    [[nodiscard]] BK_INFER_EXPORT bool load_tensor_file(
        const std::filesystem::path& path,
        TensorValue&                 tensor);

}  // namespace cvkit::infer
