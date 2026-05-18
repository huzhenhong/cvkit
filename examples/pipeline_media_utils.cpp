#include "pipeline_media_utils.h"

#include "cvkit/media/source.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <system_error>

namespace cvkit::examples
{
    namespace
    {

        [[nodiscard]] std::string escape_path_for_gstreamer(const std::filesystem::path& path)
        {
            auto        location = std::filesystem::absolute(path).string();
            std::size_t offset   = 0;
            while ((offset = location.find('\\', offset)) != std::string::npos)
            {
                location.replace(offset, 1, "\\\\");
                offset += 2;
            }

            offset = 0;
            while ((offset = location.find('"', offset)) != std::string::npos)
            {
                location.replace(offset, 1, "\\\"");
                offset += 2;
            }

            return location;
        }

        [[nodiscard]] std::string gstreamer_filesrc_pipeline(const std::filesystem::path& path)
        {
            return "filesrc location=\"" + escape_path_for_gstreamer(path)
                   + "\" ! decodebin ! videoconvert ! appsink sync=false";
        }

        [[nodiscard]] std::string gstreamer_writer_pipeline(
            const std::filesystem::path& path,
            cvkit::media::GstVideoCodec  codec)
        {
            const auto location = escape_path_for_gstreamer(path);

            if (codec == cvkit::media::GstVideoCodec::x264mp4)
            {
                return "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast ! h264parse ! mp4mux ! filesink location=\""
                       + location + "\"";
            }

            if (codec == cvkit::media::GstVideoCodec::nvh264)
            {
                return "appsrc ! videoconvert ! nvh264enc ! h264parse ! mp4mux ! filesink location=\""
                       + location + "\"";
            }

            if (codec == cvkit::media::GstVideoCodec::nvv4l2h264)
            {
                return "appsrc ! videoconvert ! nvv4l2h264enc ! h264parse ! mp4mux ! filesink location=\""
                       + location + "\"";
            }

            return "appsrc ! videoconvert ! jpegenc ! avimux ! filesink location=\"" + location + "\"";
        }

    }  // namespace

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

    bool open_video_capture(
        const std::filesystem::path& video_path,
        cvkit::media::ReaderBackend  reader_backend,
        cv::VideoCapture&            capture,
        cv::Mat&                     first_frame,
        double&                      fps)
    {
        fps = 25.0;

        if (reader_backend == cvkit::media::ReaderBackend::gstreamer)
        {
            capture.open(gstreamer_filesrc_pipeline(video_path), cv::CAP_GSTREAMER);
            if (!capture.isOpened())
            {
                std::cerr << "failed to open video with gstreamer: " << video_path << '\n';
                return false;
            }

            if (!capture.read(first_frame) || first_frame.empty())
            {
                std::cerr << "failed to read first frame with gstreamer: " << video_path << '\n';
                return false;
            }

            fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
            return true;
        }

        capture.open(video_path.string());
        if (!capture.isOpened())
        {
            std::cerr << "failed to open video: " << video_path << '\n';
            return false;
        }

        if (!capture.read(first_frame) || first_frame.empty())
        {
            std::cerr << "failed to read first frame: " << video_path << '\n';
            return false;
        }

        fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
        return true;
    }

    bool open_video_writer(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        cvkit::media::WriterBackend  writer_backend,
        cvkit::media::GstVideoCodec  gst_codec,
        double                       fps,
        const cv::Size&              frame_size,
        cv::VideoWriter&             writer,
        std::filesystem::path&       output_path)
    {
        if (writer_backend == cvkit::media::WriterBackend::ffmpeg)
        {
            std::cerr << "writer backend not implemented yet: ffmpeg\n";
            return false;
        }

        const auto output_root = ensure_output_dir(output_dir);
        output_path            = output_root / (video_path.stem().string() + "_det.mp4");

        if (writer_backend == cvkit::media::WriterBackend::gstreamer)
        {
            const bool mp4 = gst_codec == cvkit::media::GstVideoCodec::x264mp4
                             || gst_codec == cvkit::media::GstVideoCodec::nvh264
                             || gst_codec == cvkit::media::GstVideoCodec::nvv4l2h264;
            output_path = output_root / (video_path.stem().string() + (mp4 ? "_det.mp4" : "_det.avi"));
            writer.open(
                gstreamer_writer_pipeline(output_path, gst_codec),
                cv::CAP_GSTREAMER,
                0,
                fps,
                frame_size,
                true);
        }
        else
        {
            writer.open(
                output_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                frame_size);
            if (!writer.isOpened())
            {
                output_path = output_root / (video_path.stem().string() + "_det.avi");
                writer.open(
                    output_path.string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps,
                    frame_size);
            }
        }

        if (!writer.isOpened())
        {
            std::cerr << "failed to open video writer: " << output_path << '\n';
            return false;
        }

        return true;
    }

}  // namespace cvkit::examples
