#include "pipeline_support.h"
#include "example_infer_utils.h"

#include <basekit/ext/bk_cli11.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace
{

    [[nodiscard]] cvkit::infer::Backend parse_infer_backend(std::string_view value)
    {
        if (value == "tensorrt")
        {
            return cvkit::infer::Backend::tensorrt;
        }
        if (value == "onnxruntime")
        {
            return cvkit::infer::Backend::onnxruntime;
        }
        return cvkit::infer::Backend::none;
    }

}  // namespace

int main(int argc, char** argv)
{
    const auto  source_root        = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto  default_model      = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto  default_labels     = source_root / "assets" / "labels" / "coco80.txt";
    const auto  default_output_dir = source_root / "assets" / "output";

    std::string model_path  = default_model.string();
    std::string labels_path = default_labels.string();
    std::string image_path{};
    std::string video_path{};
    std::string output_dir     = default_output_dir.string();
    std::string reader_backend = "opencv";
    std::string writer_backend = "opencv";
    std::string infer_backend  = "onnxruntime";
    std::string cache_policy   = "default";
    std::string cache_dir{};
    std::string gst_codec = "jpegavi";
    std::string dump_graph_json_path{};
    bool        async_infer          = false;
    bool        print_graph          = false;
    float       confidence_threshold = 0.25F;
    float       iou_threshold        = 0.45F;
    std::size_t max_frames           = 0;

    CLI::App    app{"cvkit YOLO11 pipeline example"};
    app.add_option("--model", model_path, "Path to YOLO11 ONNX model");
    app.add_option("--labels", labels_path, "Path to class labels file");
    app.add_option("--image", image_path, "Path to input image");
    app.add_option("--video", video_path, "Path to input video");
    app.add_option("--output-dir", output_dir, "Directory for annotated outputs");
    app.add_option("--reader", reader_backend, "Reader backend: opencv, gstreamer, ffmpeg");
    app.add_option("--writer", writer_backend, "Writer backend: opencv, gstreamer, ffmpeg");
    app.add_option("--infer-backend", infer_backend, "Inference backend: onnxruntime, tensorrt");
    app.add_option("--cache-policy", cache_policy, "Model cache policy: default, disabled, read-only, rebuild");
    app.add_option("--cache-dir", cache_dir, "Optional model cache directory");
    app.add_option("--gst-codec", gst_codec, "GStreamer writer codec: jpegavi, x264mp4, nvh264, nvv4l2h264");
    app.add_flag("--async", async_infer, "Run inference through the async submit() path");
    app.add_flag("--print-graph", print_graph, "Print graph nodes, boundary, and the most recent node trace");
    app.add_option("--dump-graph-json", dump_graph_json_path, "Write graph metadata and latest trace to a JSON file");
    app.add_option("--conf", confidence_threshold, "Confidence threshold")->check(CLI::Range(0.0, 1.0));
    app.add_option("--iou", iou_threshold, "NMS IoU threshold")->check(CLI::Range(0.0, 1.0));
    app.add_option("--max-frames", max_frames, "Limit processed video frames, 0 means all frames");
    CLI11_PARSE(app, argc, argv);

    if (image_path.empty() == video_path.empty())
    {
        std::cerr << "exactly one of --image or --video is required\n";
        return 1;
    }

    cvkit::infer::Model model;
    model.set_confidence_threshold(confidence_threshold);
    model.set_iou_threshold(iou_threshold);

    if (!model.load_labels(labels_path))
    {
        std::cerr << "labels not loaded: " << labels_path << '\n';
        return 1;
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path;
    spec.labels_path  = labels_path;
    spec.backend      = parse_infer_backend(infer_backend);
    spec.task         = cvkit::infer::TaskKind::detection;
    spec.cache_policy = cvkit::examples::parse_cache_policy(cache_policy);
    spec.cache_dir    = cache_dir;
    spec.family       = "yolo11";

    if (!model.load(spec))
    {
        std::cerr << "model not loaded: " << model_path << " backend=" << infer_backend << '\n';
        return 1;
    }

    if (!image_path.empty())
    {
        return cvkit::examples::run_image(
            model,
            image_path,
            output_dir,
            cvkit::examples::parse_reader_backend(reader_backend),
            async_infer,
            print_graph,
            dump_graph_json_path);
    }

    if (!video_path.empty())
    {
        return cvkit::examples::run_video(
            model,
            video_path,
            output_dir,
            max_frames,
            cvkit::examples::parse_reader_backend(reader_backend),
            cvkit::examples::parse_writer_backend(writer_backend),
            cvkit::examples::parse_gst_codec(gst_codec),
            async_infer,
            print_graph,
            dump_graph_json_path);
    }

    std::cerr << "either --image or --video is required\n";
    return 1;
}
