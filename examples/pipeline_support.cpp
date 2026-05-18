#include "pipeline_support.h"

#include "example_graph_utils.h"
#include "example_opencv_utils.h"
#include "pipeline_media_utils.h"
#include "cvkit/core/types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
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

        cv::Mat          first_frame;
        cv::VideoCapture capture;
        double           fps = 25.0;
        if (!open_video_capture(video_path, reader_backend, capture, first_frame, fps))
        {
            return 1;
        }

        std::filesystem::path output_path;
        cv::VideoWriter       writer;
        if (!open_video_writer(video_path, output_dir, writer_backend, gst_codec, fps, first_frame.size(), writer, output_path))
        {
            return 1;
        }

        std::size_t frame_index      = 0;
        std::size_t total_detections = 0;
        cv::Mat     frame            = first_frame;
        bool        has_frame        = !frame.empty();
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
