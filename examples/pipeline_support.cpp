#include "pipeline_support.h"

#include "cvkit/core/types.h"
#include "cvkit/media/source.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace cvkit::examples
{
    namespace
    {

        [[nodiscard]] cvkit::core::Frame mat_to_frame(const cv::Mat& image, std::string source)
        {
            cvkit::core::Frame frame{};
            frame.desc.width    = image.cols;
            frame.desc.height   = image.rows;
            frame.desc.channels = image.channels();
            frame.desc.format   = image.channels() == 3 ? cvkit::core::PixelFormat::bgr8 : cvkit::core::PixelFormat::unknown;
            frame.source        = std::move(source);

            if (!image.empty())
            {
                const auto bytes = image.total() * image.elemSize();
                frame.data.assign(image.data, image.data + bytes);
            }

            return frame;
        }

        [[nodiscard]] std::vector<cvkit::core::Detection> infer_detections(
            cvkit::infer::Model& model,
            const cvkit::core::Frame& frame,
            bool async_infer)
        {
            if (!async_infer)
            {
                return model.run(frame);
            }

            cvkit::infer::TaskInput input{};
            input.add("image", frame);
            auto future = model.submit(input);
            const auto output = future.get();
            if (const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
                detections != nullptr)
            {
                return *detections;
            }

            return {};
        }

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
            return "filesrc location=\"" + escape_path_for_gstreamer(path) + "\" ! decodebin ! videoconvert ! appsink sync=false";
        }

        [[nodiscard]] std::string gstreamer_writer_pipeline(
            const std::filesystem::path& path,
            cvkit::media::GstVideoCodec  codec)
        {
            const auto location = escape_path_for_gstreamer(path);

            if (codec == cvkit::media::GstVideoCodec::x264mp4)
            {
                return "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast ! h264parse ! mp4mux ! filesink location=\"" + location + "\"";
            }

            if (codec == cvkit::media::GstVideoCodec::nvh264)
            {
                return "appsrc ! videoconvert ! nvh264enc ! h264parse ! mp4mux ! filesink location=\"" + location + "\"";
            }

            if (codec == cvkit::media::GstVideoCodec::nvv4l2h264)
            {
                return "appsrc ! videoconvert ! nvv4l2h264enc ! h264parse ! mp4mux ! filesink location=\"" + location + "\"";
            }

            return "appsrc ! videoconvert ! jpegenc ! avimux ! filesink location=\"" + location + "\"";
        }

        [[nodiscard]] cv::Scalar class_color(int class_id)
        {
            static const std::vector<cv::Scalar> palette{
                {255, 56, 56},
                {255, 157, 151},
                {255, 112, 31},
                {255, 178, 29},
                {207, 210, 49},
                {72, 249, 10},
                {146, 204, 23},
                {61, 219, 134},
                {26, 147, 52},
                {0, 212, 187},
                {44, 153, 168},
                {0, 194, 255},
                {52, 69, 147},
                {100, 115, 255},
                {0, 24, 236},
                {132, 56, 255},
                {82, 0, 133},
                {203, 56, 255},
                {255, 149, 200},
                {255, 55, 199},
            };

            if (class_id < 0)
            {
                return {0, 255, 0};
            }
            return palette[static_cast<std::size_t>(class_id) % palette.size()];
        }

        void draw_detections(cv::Mat& image, const std::vector<cvkit::core::Detection>& detections)
        {
            for (const auto& detection : detections)
            {
                const auto     color = class_color(detection.class_id);
                const cv::Rect rect(
                    static_cast<int>(std::round(detection.box.x)),
                    static_cast<int>(std::round(detection.box.y)),
                    static_cast<int>(std::round(detection.box.width)),
                    static_cast<int>(std::round(detection.box.height)));

                cv::rectangle(image, rect, color, 2);

                const auto     label     = detection.label + " " + cv::format("%.2f", detection.score);
                int            baseline  = 0;
                const auto     text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                const auto     text_y    = std::max(rect.y, text_size.height + 6);
                const cv::Rect bg(rect.x, text_y - text_size.height - 6, text_size.width + 8, text_size.height + 8);
                cv::rectangle(image, bg, color, cv::FILLED);
                cv::putText(
                    image,
                    label,
                    cv::Point(rect.x + 4, text_y - 4),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0),
                    1,
                    cv::LINE_AA);
            }
        }

        [[nodiscard]] std::filesystem::path ensure_output_dir(const std::filesystem::path& output_dir)
        {
            std::error_code ec;
            std::filesystem::create_directories(output_dir, ec);
            return output_dir;
        }

    }  // namespace

    cvkit::media::ReaderBackend parse_reader_backend(std::string_view value)
    {
        if (value == "gstreamer")
        {
            return cvkit::media::ReaderBackend::gstreamer;
        }
        if (value == "ffmpeg")
        {
            return cvkit::media::ReaderBackend::ffmpeg;
        }
        return cvkit::media::ReaderBackend::opencv;
    }

    cvkit::media::WriterBackend parse_writer_backend(std::string_view value)
    {
        if (value == "gstreamer")
        {
            return cvkit::media::WriterBackend::gstreamer;
        }
        if (value == "ffmpeg")
        {
            return cvkit::media::WriterBackend::ffmpeg;
        }
        return cvkit::media::WriterBackend::opencv;
    }

    cvkit::media::GstVideoCodec parse_gst_codec(std::string_view value)
    {
        if (value == "x264mp4")
        {
            return cvkit::media::GstVideoCodec::x264mp4;
        }
        if (value == "nvh264")
        {
            return cvkit::media::GstVideoCodec::nvh264;
        }
        if (value == "nvv4l2h264")
        {
            return cvkit::media::GstVideoCodec::nvv4l2h264;
        }
        return cvkit::media::GstVideoCodec::jpegavi;
    }

    int run_image(
        cvkit::infer::Model&         model,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_dir,
        cvkit::media::ReaderBackend  reader_backend,
        bool                         async_infer)
    {
        cv::Mat image;
        if (reader_backend == cvkit::media::ReaderBackend::gstreamer)
        {
            cvkit::media::Source        source;
            cvkit::media::SourceOptions source_options{};
            source_options.uri     = image_path.string();
            source_options.backend = reader_backend;
            if (!source.open(std::move(source_options)))
            {
                std::cerr << "failed to open image with gstreamer: " << image_path << '\n';
                return 1;
            }

            cvkit::core::Frame frame{};
            if (!source.read(frame))
            {
                std::cerr << "failed to read image with gstreamer: " << image_path << '\n';
                return 1;
            }

            image = cv::Mat(
                        frame.desc.height,
                        frame.desc.width,
                        frame.desc.channels == 3 ? CV_8UC3 : CV_8UC1,
                        frame.data.data())
                        .clone();
        }
        else
        {
            image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        }

        if (image.empty())
        {
            std::cerr << "failed to read image: " << image_path << '\n';
            return 1;
        }

        auto detections = infer_detections(model, mat_to_frame(image, image_path.string()), async_infer);
        draw_detections(image, detections);

        const auto output_root = ensure_output_dir(output_dir);
        const auto output_path = output_root / (image_path.stem().string() + "_det" + image_path.extension().string());
        if (!cv::imwrite(output_path.string(), image))
        {
            std::cerr << "failed to write image: " << output_path << '\n';
            return 1;
        }

        std::cout
            << "image=" << image_path
            << " async=" << (async_infer ? "true" : "false")
            << " detections=" << detections.size()
            << " output=" << output_path
            << '\n';
        return 0;
    }

    int run_video(
        cvkit::infer::Model&         model,
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        std::size_t                  max_frames,
        cvkit::media::ReaderBackend  reader_backend,
        cvkit::media::WriterBackend  writer_backend,
        cvkit::media::GstVideoCodec  gst_codec,
        bool                         async_infer)
    {
        if (writer_backend == cvkit::media::WriterBackend::ffmpeg)
        {
            std::cerr << "writer backend not implemented yet: ffmpeg\n";
            return 1;
        }

        cv::Mat          first_frame;
        cv::VideoCapture capture;
        double           fps = 25.0;

        if (reader_backend == cvkit::media::ReaderBackend::gstreamer)
        {
            capture.open(gstreamer_filesrc_pipeline(video_path), cv::CAP_GSTREAMER);
            if (!capture.isOpened())
            {
                std::cerr << "failed to open video with gstreamer: " << video_path << '\n';
                return 1;
            }

            if (!capture.read(first_frame) || first_frame.empty())
            {
                std::cerr << "failed to read first frame with gstreamer: " << video_path << '\n';
                return 1;
            }
            fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
        }
        else
        {
            capture.open(video_path.string());
            if (!capture.isOpened())
            {
                std::cerr << "failed to open video: " << video_path << '\n';
                return 1;
            }

            if (!capture.read(first_frame) || first_frame.empty())
            {
                std::cerr << "failed to read first frame: " << video_path << '\n';
                return 1;
            }

            fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
        }

        const auto      output_root = ensure_output_dir(output_dir);
        auto            output_path = output_root / (video_path.stem().string() + "_det.mp4");
        cv::VideoWriter writer;
        if (writer_backend == cvkit::media::WriterBackend::gstreamer)
        {
            const bool mp4 = gst_codec == cvkit::media::GstVideoCodec::x264mp4 || gst_codec == cvkit::media::GstVideoCodec::nvh264 || gst_codec == cvkit::media::GstVideoCodec::nvv4l2h264;
            output_path    = output_root / (video_path.stem().string() + (mp4 ? "_det.mp4" : "_det.avi"));
            writer.open(
                gstreamer_writer_pipeline(output_path, gst_codec),
                cv::CAP_GSTREAMER,
                0,
                fps,
                first_frame.size(),
                true);
        }
        else
        {
            writer.open(
                output_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                first_frame.size());
            if (!writer.isOpened())
            {
                output_path = output_root / (video_path.stem().string() + "_det.avi");
                writer.open(
                    output_path.string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps,
                    first_frame.size());
            }
        }

        if (!writer.isOpened())
        {
            std::cerr << "failed to open video writer: " << output_path << '\n';
            return 1;
        }

        std::size_t frame_index      = 0;
        std::size_t total_detections = 0;
        cv::Mat     frame            = first_frame;
        bool        has_frame        = !frame.empty();
        do
        {
            auto detections = infer_detections(model, mat_to_frame(frame, video_path.string()), async_infer);
            total_detections += detections.size();
            draw_detections(frame, detections);
            writer.write(frame);
            ++frame_index;
            if (max_frames > 0 && frame_index >= max_frames)
            {
                break;
            }

            has_frame = capture.read(frame) && !frame.empty();
        } while (has_frame);

        std::cout
            << "video=" << video_path
            << " async=" << (async_infer ? "true" : "false")
            << " frames=" << frame_index
            << " detections=" << total_detections
            << " output=" << output_path
            << '\n';
        return 0;
    }

}  // namespace cvkit::examples
