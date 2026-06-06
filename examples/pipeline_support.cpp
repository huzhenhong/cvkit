#include "pipeline_support.h"

#include "example_graph_utils.h"
#include "example_opencv_utils.h"
#include "pipeline_media_utils.h"
#include "cvkit/core/types.h"
#include "cvkit/media/source.h"
#include "cvkit/media/writer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::examples
{
    namespace
    {

        [[nodiscard]] std::vector<cvkit::core::Detection> infer_detections(
            cvkit::infer::Model&            model,
            const cvkit::infer::ImageValue& image,
            bool                            async_infer)
        {
            if (!async_infer)
            {
                return model.run_detection(image.frame);
            }

            cvkit::infer::TaskInput input{};
            input.add("image", image);
            auto       future = model.submit(input);
            const auto output = future.get();
            if (const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
                detections != nullptr)
            {
                return *detections;
            }

            return {};
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

        [[nodiscard]] cv::Mat frame_to_mat(const cvkit::core::Frame& frame)
        {
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels != 3 ||
                frame.desc.format != cvkit::core::PixelFormat::bgr8 || frame.data.empty())
            {
                return {};
            }

            return cv::Mat(
                       frame.desc.height,
                       frame.desc.width,
                       CV_8UC3,
                       const_cast<std::uint8_t*>(frame.data.data()))
                .clone();
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
        bool                         async_infer,
        bool                         print_graph,
        const std::filesystem::path& dump_graph_json_path)
    {
        if (print_graph)
        {
            cvkit::examples::print_graph_info(std::cout, model);
        }

        cv::Mat image;
        if (!read_image(image_path, reader_backend, image))
        {
            std::cerr << "failed to read image: " << image_path << '\n';
            return 1;
        }

        auto detections = infer_detections(model, mat_to_image_value(image, image_path.string()), async_infer);
        if (print_graph)
        {
            cvkit::examples::print_graph_trace(std::cout, model);
        }
        if (!cvkit::examples::dump_graph_json(model, async_infer, dump_graph_json_path))
        {
            return 1;
        }
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
        bool                         async_infer,
        bool                         print_graph,
        const std::filesystem::path& dump_graph_json_path)
    {
        if (print_graph)
        {
            cvkit::examples::print_graph_info(std::cout, model);
        }

        cvkit::media::SourceOptions source_options{};
        source_options.uri     = video_path.string();
        source_options.backend = reader_backend;

        cvkit::media::Source source;
        if (!source.open(std::move(source_options)))
        {
            std::cerr << "failed to open video source: " << source.status_message() << '\n';
            return 1;
        }

        cvkit::core::Frame first_source_frame{};
        if (!source.read(first_source_frame))
        {
            std::cerr << "failed to read first video frame: " << source.status_message() << '\n';
            return 1;
        }

        auto first_frame = frame_to_mat(first_source_frame);
        if (first_frame.empty())
        {
            std::cerr << "failed to convert first video frame to BGR image\n";
            return 1;
        }

        const auto source_info = source.info();
        const auto fps         = std::max(1.0, source_info.fps);

        const auto            output_root = ensure_output_dir(output_dir);
        const bool            mp4_output  = writer_backend == cvkit::media::WriterBackend::gstreamer &&
                                 (gst_codec == cvkit::media::GstVideoCodec::x264mp4 ||
                                  gst_codec == cvkit::media::GstVideoCodec::nvh264);
        const auto            output_path = output_root / (video_path.stem().string() + (mp4_output ? "_det.mp4" : "_det.avi"));
        cvkit::media::Writer  writer;
        cvkit::media::WriterOptions writer_options{};
        writer_options.uri     = output_path.string();
        writer_options.backend = writer_backend;
        writer_options.gst_codec = gst_codec;
        writer_options.width   = first_frame.cols;
        writer_options.height  = first_frame.rows;
        writer_options.fps     = fps;
        if (!writer.open(std::move(writer_options)))
        {
            std::cerr << "failed to open video writer: " << writer.status_message() << '\n';
            return 1;
        }

        std::size_t frame_index      = 0;
        std::size_t total_detections = 0;
        cv::Mat     frame            = first_frame;
        bool        has_frame        = !frame.empty();
        cvkit::core::Frame source_frame = std::move(first_source_frame);
        do
        {
            auto detections = infer_detections(model, mat_to_image_value(frame, video_path.string()), async_infer);
            if (print_graph && frame_index == 0)
            {
                cvkit::examples::print_graph_trace(std::cout, model);
            }
            if (frame_index == 0 && !cvkit::examples::dump_graph_json(model, async_infer, dump_graph_json_path))
            {
                return 1;
            }
            total_detections += detections.size();
            draw_detections(frame, detections);
            const auto output_frame = mat_to_frame(frame, video_path.string());
            if (!writer.write(output_frame))
            {
                std::cerr << "failed to write video frame: " << writer.status_message() << '\n';
                return 1;
            }
            ++frame_index;
            if (max_frames > 0 && frame_index >= max_frames)
            {
                break;
            }

            source_frame = {};
            has_frame    = source.read(source_frame);
            if (has_frame)
            {
                frame = frame_to_mat(source_frame);
                has_frame = !frame.empty();
            }
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
