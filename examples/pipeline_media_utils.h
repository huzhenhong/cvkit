#pragma once

#include "cvkit/media/options.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>

namespace cvkit::examples
{

    [[nodiscard]] std::filesystem::path ensure_output_dir(const std::filesystem::path& output_dir);

    [[nodiscard]] bool                  read_image(
        const std::filesystem::path& image_path,
        cvkit::media::ReaderBackend  reader_backend,
        cv::Mat&                     image);

    [[nodiscard]] bool open_video_capture(
        const std::filesystem::path& video_path,
        cvkit::media::ReaderBackend  reader_backend,
        cv::VideoCapture&            capture,
        cv::Mat&                     first_frame,
        double&                      fps);

    [[nodiscard]] bool open_video_writer(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        cvkit::media::WriterBackend  writer_backend,
        cvkit::media::GstVideoCodec  gst_codec,
        double                       fps,
        const cv::Size&              frame_size,
        cv::VideoWriter&             writer,
        std::filesystem::path&       output_path);

}  // namespace cvkit::examples
