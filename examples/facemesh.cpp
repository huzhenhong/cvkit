#include <basekit/ext/bk_cli11.h>

#include "example_graph_utils.h"
#include "example_infer_utils.h"
#include "example_opencv_utils.h"

#include "cvkit/infer/model.h"

#include <opencv2/imgcodecs.hpp>

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
    std::string model_path{};
    std::string image_path{};
    std::string infer_backend = "onnxruntime";
    std::string cache_policy  = "default";
    std::string cache_dir{};
    std::string dump_graph_json_path{};
    bool        async_infer = false;
    bool        print_graph = false;

    CLI::App app{"cvkit facemesh example"};
    app.add_option("--model", model_path, "Path to facemesh ONNX model")->required();
    app.add_option("--image", image_path, "Path to input image")->required();
    app.add_option("--infer-backend", infer_backend, "Inference backend: onnxruntime, tensorrt");
    app.add_option("--cache-policy", cache_policy, "Model cache policy: default, disabled, read-only, rebuild");
    app.add_option("--cache-dir", cache_dir, "Optional model cache directory");
    app.add_option("--dump-graph-json", dump_graph_json_path, "Write graph metadata and latest trace to a JSON file");
    app.add_flag("--async", async_infer, "Run facemesh through the async submit() path");
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
    spec.task         = cvkit::infer::TaskKind::facemesh;
    spec.family       = "facemesh";
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

    const auto* landmarks = output.find<cvkit::infer::KeypointsValue>("landmarks");
    const auto* scores    = output.find<std::vector<float>>("scores");
    if (landmarks == nullptr)
    {
        std::cerr << "facemesh output missing landmarks\n";
        return 1;
    }

    std::cout
        << "image=" << image_path
        << " backend=" << infer_backend
        << " async=" << (async_infer ? "true" : "false")
        << " landmark_count=" << landmarks->points.size();
    if (scores != nullptr)
    {
        std::cout << " score_count=" << scores->size();
    }
    std::cout << '\n';

    return 0;
}
