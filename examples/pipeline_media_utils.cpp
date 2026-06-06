#include "pipeline_media_utils.h"

#include "cvkit/media/source.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <system_error>

namespace cvkit::examples
{
    std::filesystem::path ensure_output_dir(const std::filesystem::path& output_dir)
    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        return output_dir;
    }

    bool read_image(
        const std::filesystem::path& image_path,
        cvkit::media::ReaderBackend  reader_backend,
        cv::Mat&                     image)
    {
        if (reader_backend == cvkit::media::ReaderBackend::gstreamer)
        {
            cvkit::media::Source        source;
            cvkit::media::SourceOptions source_options{};
            source_options.uri     = image_path.string();
            source_options.backend = reader_backend;
            if (!source.open(std::move(source_options)))
            {
                std::cerr << "failed to open image with gstreamer: " << image_path << '\n';
                return false;
            }

            cvkit::core::Frame frame{};
            if (!source.read(frame))
            {
                std::cerr << "failed to read image with gstreamer: " << image_path << '\n';
                return false;
            }

            image = cv::Mat(
                        frame.desc.height,
                        frame.desc.width,
                        frame.desc.channels == 3 ? CV_8UC3 : CV_8UC1,
                        frame.data.data())
                        .clone();
            return !image.empty();
        }

        image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        return !image.empty();
    }

}  // namespace cvkit::examples
