#include <basekit/ext/bk_cli11.h>

#include "example_graph_utils.h"
#include "example_infer_utils.h"
#include "example_opencv_utils.h"

#include "cvkit/infer/model.h"

#include <opencv2/imgcodecs.hpp>

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

}  // namespace

int main(int argc, char** argv)
{
    const auto  source_root        = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto  default_output_dir = source_root / "assets" / "output";

    std::string model_path{};
    std::string image_path{};
    std::string output_dir    = default_output_dir.string();
    std::string infer_backend = "onnxruntime";
    std::string cache_policy  = "default";
    std::string cache_dir{};
    std::string dump_graph_json_path{};
    bool        async_infer = false;
    bool        print_graph = false;

    CLI::App app{"cvkit segmentation example"};
    app.add_option("--model", model_path, "Path to segmentation ONNX model")->required();
    app.add_option("--image", image_path, "Path to input image")->required();
    app.add_option("--output-dir", output_dir, "Directory for mask output");
    app.add_option("--infer-backend", infer_backend, "Inference backend: onnxruntime, tensorrt");
    app.add_option("--cache-policy", cache_policy, "Model cache policy: default, disabled, read-only, rebuild");
    app.add_option("--cache-dir", cache_dir, "Optional model cache directory");
    app.add_option("--dump-graph-json", dump_graph_json_path, "Write graph metadata and latest trace to a JSON file");
    app.add_flag("--async", async_infer, "Run segmentation through the async submit() path");
    app.add_flag("--print-graph", print_graph, "Print graph nodes, boundary, and the most recent node trace");
    CLI11_PARSE(app, argc, argv);

    const auto image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "failed to read image: " << image_path << '\n';
        return 1;
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path;
    spec.backend      = parse_infer_backend(infer_backend);
    spec.task         = cvkit::infer::TaskKind::segmentation;
    spec.family       = "segmentation";
    spec.cache_policy = cvkit::examples::parse_cache_policy(cache_policy);
    spec.cache_dir    = cache_dir;

    cvkit::infer::Model model;
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

    const auto* mask_value = output.find<cvkit::infer::MaskValue>("mask");
    const auto* logits     = output.find<cvkit::infer::TensorValue>("logits");
    if (mask_value == nullptr)
    {
        std::cerr << "segmentation output missing mask\n";
        return 1;
    }

    const auto mask = cvkit::examples::frame_mask_to_mat(mask_value->frame);
    if (mask.empty())
    {
        std::cerr << "segmentation mask was empty\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const auto output_path =
        std::filesystem::path(output_dir) / (std::filesystem::path(image_path).stem().string() + "_seg_mask.png");
    if (!cv::imwrite(output_path.string(), mask))
    {
        std::cerr << "failed to write mask image: " << output_path << '\n';
        return 1;
    }

    std::cout
        << "image=" << image_path
        << " backend=" << infer_backend
        << " async=" << (async_infer ? "true" : "false")
        << " mask_output=" << output_path;
    if (logits != nullptr)
    {
        std::cout << " logits_shape=";
        for (std::size_t index = 0; index < logits->shape.size(); ++index)
        {
            if (index > 0)
            {
                std::cout << 'x';
            }
            std::cout << logits->shape[index];
        }
    }
    std::cout << '\n';

    return 0;
}
