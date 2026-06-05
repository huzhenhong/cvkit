#include <basekit/ext/bk_cli11.h>

#include "example_graph_utils.h"
#include "example_infer_utils.h"
#include "example_opencv_utils.h"

#include "cvkit/infer/model.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace
{

    cvkit::infer::Backend parse_infer_backend(std::string_view value)
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

    void draw_faces(cv::Mat& image, const std::vector<cvkit::core::Detection>& detections)
    {
        for (const auto& detection : detections)
        {
            const cv::Rect rect(
                static_cast<int>(std::round(detection.box.x)),
                static_cast<int>(std::round(detection.box.y)),
                static_cast<int>(std::round(detection.box.width)),
                static_cast<int>(std::round(detection.box.height)));
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

            for (const auto& point : detection.keypoints)
            {
                cv::circle(
                    image,
                    cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))),
                    2,
                    cv::Scalar(0, 0, 255),
                    cv::FILLED);
            }
        }
    }

}  // namespace

int main(int argc, char** argv)
{
    const auto  source_root        = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto  default_output_dir = source_root / "assets" / "output";

    std::string model_path{};
    std::string image_path{};
    std::string output_dir    = default_output_dir.string();
    std::string infer_backend = "onnxruntime";
    std::string family        = "scrfd";
    std::string cache_policy  = "default";
    std::string cache_dir{};
    std::string dump_graph_json_path{};
    bool        async_infer = false;
    bool        print_graph = false;
    float       confidence_threshold = 0.5F;
    float       iou_threshold        = 0.4F;
    int         tile_width           = 0;
    int         tile_height          = 0;
    int         tile_overlap_x       = 0;
    int         tile_overlap_y       = 0;

    CLI::App app{"cvkit face detection example"};
    app.add_option("--model", model_path, "Path to face detection ONNX model")->required();
    app.add_option("--image", image_path, "Path to input image")->required();
    app.add_option("--output-dir", output_dir, "Directory for annotated output");
    app.add_option("--infer-backend", infer_backend, "Inference backend: onnxruntime, tensorrt");
    app.add_option("--family", family, "Model family: scrfd, scrfd_raw_bgr");
    app.add_option("--cache-policy", cache_policy, "Model cache policy: default, disabled, read-only, rebuild");
    app.add_option("--cache-dir", cache_dir, "Optional model cache directory");
    app.add_option("--dump-graph-json", dump_graph_json_path, "Write graph metadata and latest trace to a JSON file");
    app.add_option("--conf", confidence_threshold, "Confidence threshold")->check(CLI::Range(0.0, 1.0));
    app.add_option("--iou", iou_threshold, "NMS IoU threshold")->check(CLI::Range(0.0, 1.0));
    app.add_option("--tile-width", tile_width, "Enable tiled inference with this tile width; 0 disables tiling")->check(CLI::NonNegativeNumber);
    app.add_option("--tile-height", tile_height, "Enable tiled inference with this tile height; 0 disables tiling")->check(CLI::NonNegativeNumber);
    app.add_option("--tile-overlap-x", tile_overlap_x, "Tiled inference horizontal overlap in pixels")->check(CLI::NonNegativeNumber);
    app.add_option("--tile-overlap-y", tile_overlap_y, "Tiled inference vertical overlap in pixels")->check(CLI::NonNegativeNumber);
    app.add_flag("--async", async_infer, "Run face detection through the async submit() path");
    app.add_flag("--print-graph", print_graph, "Print graph nodes, boundary, and the most recent node trace");
    CLI11_PARSE(app, argc, argv);

    auto image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "failed to read image: " << image_path << '\n';
        return 1;
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path;
    spec.backend      = parse_infer_backend(infer_backend);
    spec.task         = cvkit::infer::TaskKind::face_detection;
    spec.family       = family;
    spec.cache_policy = cvkit::examples::parse_cache_policy(cache_policy);
    spec.cache_dir    = cache_dir;

    cvkit::infer::Model model;
    model.set_confidence_threshold(confidence_threshold);
    model.set_iou_threshold(iou_threshold);
    if (tile_width > 0 || tile_height > 0)
    {
        cvkit::infer::TileOptions tile_options{};
        tile_options.enabled     = true;
        tile_options.tile_width  = tile_width > 0 ? tile_width : tile_height;
        tile_options.tile_height = tile_height > 0 ? tile_height : tile_width;
        tile_options.overlap_x   = tile_overlap_x;
        tile_options.overlap_y   = tile_overlap_y;
        model.set_tile_options(tile_options);
    }
    if (!model.load(spec))
    {
        std::cerr << "model not loaded: " << model_path << " backend=" << infer_backend << '\n';
        return 1;
    }

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::examples::mat_to_image_value(image, image_path));

    if (print_graph)
    {
        cvkit::examples::print_graph_info(std::cout, model);
    }

    const auto output =
        async_infer ? model.submit(input).get() : model.run_sync(input);

    if (print_graph)
    {
        cvkit::examples::print_graph_trace(std::cout, model);
    }
    if (!cvkit::examples::dump_graph_json(model, async_infer, dump_graph_json_path))
    {
        return 1;
    }

    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    if (detections == nullptr)
    {
        std::cerr << "face detection output missing detections\n";
        return 1;
    }

    draw_faces(image, *detections);

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const auto output_path =
        std::filesystem::path(output_dir) / (std::filesystem::path(image_path).stem().string() + "_face_det.png");
    if (!cv::imwrite(output_path.string(), image))
    {
        std::cerr << "failed to write image: " << output_path << '\n';
        return 1;
    }

    std::size_t keypoint_count = 0;
    for (const auto& detection : *detections)
    {
        keypoint_count += detection.keypoints.size();
    }

    std::cout
        << "image=" << image_path
        << " backend=" << infer_backend
        << " async=" << (async_infer ? "true" : "false")
        << " faces=" << detections->size()
        << " keypoints=" << keypoint_count
        << " output=" << output_path
        << '\n';

    return 0;
}
