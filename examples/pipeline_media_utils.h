#pragma once

#include "cvkit/media/options.h"

#include <opencv2/core.hpp>

#include <filesystem>

namespace cvkit::examples
{

    [[nodiscard]] std::filesystem::path ensure_output_dir(const std::filesystem::path& output_dir);

    [[nodiscard]] bool                  read_image(
        const std::filesystem::path& image_path,
        cvkit::media::ReaderBackend  reader_backend,
        cv::Mat&                     image);

}  // namespace cvkit::examples
