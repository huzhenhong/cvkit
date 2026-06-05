#include "example_infer_utils.h"

#include "cvkit/infer/model.h"
#include "cvkit/infer/task_io.h"
#include "cvkit/media/source.h"

#include <basekit/ext/bk_cli11.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if defined(CVKIT_VIDEO_GPU_DETECTION_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

    cvkit::infer::ModelSpec make_detection_spec(
        const std::string& model_path,
        const std::string& labels_path,
        const std::string& cache_dir)
    {
        cvkit::infer::ModelSpec spec{};
        spec.model_path   = model_path;
        spec.labels_path  = labels_path;
        spec.backend      = cvkit::infer::Backend::tensorrt;
        spec.task         = cvkit::infer::TaskKind::detection;
        spec.family       = "yolo11";
        spec.cache_policy = cvkit::infer::CachePolicy::default_policy;
        spec.cache_dir    = cache_dir;
        spec.device       = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
        spec.tensorrt_profiles.push_back({
            .input_name = "images",
            .shape =
                {
                    .min = {1, 3, 320, 320},
                    .opt = {1, 3, 640, 640},
                    .max = {1, 3, 1280, 1280},
                },
        });
        return spec;
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
            const auto color = class_color(detection.class_id);
            const cv::Rect rect(
                static_cast<int>(std::round(detection.box.x)),
                static_cast<int>(std::round(detection.box.y)),
                static_cast<int>(std::round(detection.box.width)),
                static_cast<int>(std::round(detection.box.height)));

            cv::rectangle(image, rect, color, 2);

            const auto label = detection.label.empty()
                                   ? cv::format("%.2f", detection.score)
                                   : detection.label + " " + cv::format("%.2f", detection.score);
            int        baseline = 0;
            const auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            const auto text_y = std::max(rect.y, text_size.height + 6);
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

    [[nodiscard]] cv::Mat download_nv12_device_frame(const cvkit::core::DeviceFrame& frame, std::string* error)
    {
        if (!frame.valid() || frame.memory_device != cvkit::core::MemoryDevice::cuda ||
            frame.desc.format != cvkit::core::PixelFormat::nv12 || frame.stride_bytes == 0U)
        {
            if (error != nullptr)
            {
                *error = "only CUDA NV12 DeviceFrame output can be downloaded by this example";
            }
            return {};
        }

#if !defined(CVKIT_VIDEO_GPU_DETECTION_WITH_CUDA_RUNTIME)
        if (error != nullptr)
        {
            *error = "example was built without CUDA runtime download support";
        }
        return {};
#else
        const auto width = frame.desc.width;
        const auto height = frame.desc.height;
        const auto stride = frame.stride_bytes;
        if (width <= 0 || height <= 0 || stride < static_cast<std::size_t>(width))
        {
            if (error != nullptr)
            {
                *error = "invalid NV12 frame geometry";
            }
            return {};
        }

        const auto y_bytes = stride * static_cast<std::size_t>(height);
        const auto uv_height = (height + 1) / 2;
        const auto uv_bytes = stride * static_cast<std::size_t>(uv_height);
        const auto required_bytes = y_bytes + uv_bytes;
        if (frame.bytes < required_bytes)
        {
            if (error != nullptr)
            {
                *error = "NV12 frame storage is smaller than expected";
            }
            return {};
        }

        cv::Mat nv12(height + uv_height, width, CV_8UC1);
        auto* source_y = reinterpret_cast<const void*>(frame.data);
        auto* source_uv = reinterpret_cast<const void*>(frame.data + y_bytes);
        if (cudaMemcpy2D(
                nv12.data,
                static_cast<std::size_t>(nv12.step[0]),
                source_y,
                stride,
                static_cast<std::size_t>(width),
                static_cast<std::size_t>(height),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            if (error != nullptr)
            {
                *error = "failed to download NV12 Y plane";
            }
            return {};
        }
        if (cudaMemcpy2D(
                nv12.data + static_cast<std::size_t>(height) * nv12.step[0],
                static_cast<std::size_t>(nv12.step[0]),
                source_uv,
                stride,
                static_cast<std::size_t>(width),
                static_cast<std::size_t>(uv_height),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            if (error != nullptr)
            {
                *error = "failed to download NV12 UV plane";
            }
            return {};
        }

        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        return bgr;
#endif
    }

}  // namespace

int main(int argc, char** argv)
{
    const auto source_root   = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto default_video  = source_root / "assets" / "video" / "test.mp4";
    const auto default_model  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto default_labels = source_root / "assets" / "labels" / "coco80.txt";

    std::string video_path           = default_video.string();
    std::string model_path           = default_model.string();
    std::string labels_path          = default_labels.string();
    std::string cache_dir            = {};
    std::string output_image_path    = {};
    int         cuda_device          = 0;
    std::size_t max_frames           = 30;
    std::size_t dump_frame_index     = 0;
    float       confidence_threshold = 0.25F;
    float       iou_threshold        = 0.45F;

    CLI::App app{"cvkit GPU video detection smoke example"};
    app.add_option("--video", video_path, "Path to input H.264 MP4 video");
    app.add_option("--model", model_path, "Path to YOLO ONNX model");
    app.add_option("--labels", labels_path, "Path to class labels file");
    app.add_option("--cache-dir", cache_dir, "Optional TensorRT engine cache directory");
    app.add_option("--output-image", output_image_path, "Optional annotated image output path; explicitly downloads one NV12 frame");
    app.add_option("--cuda-device", cuda_device, "CUDA device index inside the current process");
    app.add_option("--max-frames", max_frames, "Maximum frames to process");
    app.add_option("--dump-frame", dump_frame_index, "Frame index to download when --output-image is set");
    app.add_option("--conf", confidence_threshold, "Confidence threshold")->check(CLI::Range(0.0, 1.0));
    app.add_option("--iou", iou_threshold, "NMS IoU threshold")->check(CLI::Range(0.0, 1.0));
    CLI11_PARSE(app, argc, argv);

    cvkit::infer::Model model;
    model.set_confidence_threshold(confidence_threshold);
    model.set_iou_threshold(iou_threshold);
    if (!model.load(make_detection_spec(model_path, labels_path, cache_dir)))
    {
        std::cerr << "failed to load TensorRT detection model: " << model_path << '\n';
        return 1;
    }

    cvkit::media::SourceOptions source_options{};
    source_options.uri               = video_path;
    source_options.backend           = cvkit::media::ReaderBackend::gstreamer;
    source_options.output_memory     = cvkit::media::SourceMemory::cuda;
    source_options.cuda_device_index = cuda_device;

    cvkit::media::Source source;
    if (!source.open(std::move(source_options)))
    {
        std::cerr << "failed to open CUDA video source: " << source.status_message() << '\n';
        return 1;
    }

    std::size_t frame_count     = 0;
    std::size_t detection_count = 0;
    bool        wrote_output_image = false;
    while (max_frames == 0 || frame_count < max_frames)
    {
        cvkit::core::DeviceFrame frame{};
        if (!source.read(frame))
        {
            break;
        }
        if (!frame.valid())
        {
            std::cerr << "invalid device frame at index " << frame_count << '\n';
            return 1;
        }

        cvkit::infer::TaskInput input{};
        input.add("image", cvkit::infer::image_value_from_device_frame(frame));

        const auto output = model.run_sync(input);
        const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
        if (detections == nullptr)
        {
            std::cerr << "detection output missing at frame " << frame_count << '\n';
            return 1;
        }

        detection_count += detections->size();
        if (!output_image_path.empty() && frame_count == dump_frame_index)
        {
            std::string error;
            auto        image = download_nv12_device_frame(frame, &error);
            if (image.empty())
            {
                std::cerr << "failed to download output frame: " << error << '\n';
                return 1;
            }
            draw_detections(image, *detections);
            if (!cv::imwrite(output_image_path, image))
            {
                std::cerr << "failed to write output image: " << output_image_path << '\n';
                return 1;
            }
            wrote_output_image = true;
        }
        ++frame_count;
    }

    if (!output_image_path.empty() && !wrote_output_image)
    {
        std::cerr << "requested dump frame was not reached: " << dump_frame_index << '\n';
        return 1;
    }

    std::cout
        << "video=" << video_path
        << " frames=" << frame_count
        << " detections=" << detection_count
        << " memory=cuda"
        << " format=nv12"
        << " output_image=" << (wrote_output_image ? output_image_path : "")
        << '\n';
    return frame_count > 0 ? 0 : 1;
}
