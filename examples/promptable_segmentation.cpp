#include <basekit/ext/bk_cli11.h>

#include "example_opencv_utils.h"
#include "promptable_segmentation_support.h"

#include "cvkit/infer/model.h"
#include "cvkit/infer/tensor_io.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    namespace promptable = cvkit::examples::promptable;

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto default_encoder = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto default_decoder = source_root / "assets" / "models" / "efficient_sam_vitt_decoder.sim.onnx";
    const auto default_output_dir = source_root / "assets" / "output";

    std::string encoder_path = default_encoder.string();
    std::string decoder_path = default_decoder.string();
    std::string mode_name_value = "combined";
    std::string embeddings_path{};
    std::string image_path{};
    std::string output_dir = default_output_dir.string();
    std::string cache_policy = "default";
    std::string cache_dir{};
    std::string dump_graph_json_path{};
    bool        async_infer = false;
    bool        print_graph = false;
    float       point_x = -1.0F;
    float       point_y = -1.0F;
    float       point_label = 1.0F;
    float       box_x = 0.0F;
    float       box_y = 0.0F;
    float       box_w = 0.0F;
    float       box_h = 0.0F;
    bool        use_box = false;

    CLI::App app{"cvkit EfficientSAM promptable segmentation example"};
    app.add_option("--mode", mode_name_value, "Run mode: combined, encoder, decoder");
    app.add_option("--encoder", encoder_path, "Path to EfficientSAM encoder ONNX model");
    app.add_option("--decoder", decoder_path, "Path to EfficientSAM decoder ONNX model");
    app.add_option("--embeddings", embeddings_path, "Path to embedding tensor file for encoder output or decoder input");
    app.add_option("--image", image_path, "Path to input image")->required();
    app.add_option("--output-dir", output_dir, "Directory for mask and overlay outputs");
    app.add_option("--cache-policy", cache_policy, "Model cache policy: default, disabled, read-only, rebuild");
    app.add_option("--cache-dir", cache_dir, "Optional model cache directory");
    app.add_option("--dump-graph-json", dump_graph_json_path, "Write graph metadata and latest trace to a JSON file");
    app.add_flag("--async", async_infer, "Run segmentation through the async submit() path");
    app.add_flag("--print-graph", print_graph, "Print graph nodes, boundary, and the most recent node trace");
    app.add_option("--point-x", point_x, "Prompt point x in image coordinates");
    app.add_option("--point-y", point_y, "Prompt point y in image coordinates");
    app.add_option("--point-label", point_label, "Prompt point label, usually 1 for positive and 0 for negative");
    app.add_option("--box-x", box_x, "Prompt box x");
    app.add_option("--box-y", box_y, "Prompt box y");
    app.add_option("--box-w", box_w, "Prompt box width");
    app.add_option("--box-h", box_h, "Prompt box height");
    app.add_flag("--use-box", use_box, "Use the provided box prompt");
    CLI11_PARSE(app, argc, argv);

    const auto mode = promptable::parse_mode(mode_name_value);
    const auto image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "failed to read image: " << image_path << '\n';
        return 1;
    }

    if (embeddings_path.empty())
    {
        const auto stem = std::filesystem::path(image_path).stem().string();
        embeddings_path = (std::filesystem::path(output_dir) / (stem + "_sam_embeddings.bin")).string();
    }

    cvkit::infer::ModelSpec spec{};
    spec.backend = cvkit::infer::Backend::onnxruntime;
    spec.task = cvkit::infer::TaskKind::promptable_segmentation;
    spec.cache_policy = promptable::parse_cache_policy(cache_policy);
    spec.cache_dir = cache_dir;
    switch (mode)
    {
        case promptable::Mode::encoder:
            spec.model_path = encoder_path;
            spec.family = "efficient_sam_encoder";
            break;
        case promptable::Mode::decoder:
            spec.model_path = decoder_path;
            spec.family = "efficient_sam_decoder";
            break;
        case promptable::Mode::combined:
        default:
            spec.model_path = encoder_path;
            spec.aux_model_path = decoder_path;
            spec.family = "efficient_sam";
            break;
    }

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        std::cerr << "failed to load efficient_sam model(s)\n";
        return 1;
    }

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::examples::mat_to_image_value(image, image_path));
    if (mode == promptable::Mode::decoder)
    {
        cvkit::infer::TensorValue embeddings{};
        if (!cvkit::infer::load_tensor_file(embeddings_path, embeddings))
        {
            std::cerr << "failed to load embedding tensor: " << embeddings_path << '\n';
            return 1;
        }
        input.add("image_embeddings", std::move(embeddings));
    }
    if (point_x >= 0.0F && point_y >= 0.0F)
    {
        input.add("points", std::vector<cvkit::core::Point2f>{{point_x, point_y}});
        input.add("point_labels", std::vector<float>{point_label});
    }
    if (use_box)
    {
        input.add("box", cvkit::core::BBox{box_x, box_y, box_w, box_h});
    }

    if (print_graph)
    {
        promptable::print_graph_info(model);
    }

    cvkit::infer::TaskOutput output =
        async_infer ? model.submit(input).get() : model.run_sync(input);

    if (print_graph)
    {
        promptable::print_graph_trace(model);
    }
    if (!promptable::dump_graph_json(model, async_infer, mode, embeddings_path, dump_graph_json_path))
    {
        return 1;
    }

    const auto* embeddings = output.find<cvkit::infer::TensorValue>("image_embeddings");
    if (mode == promptable::Mode::encoder)
    {
        if (embeddings == nullptr)
        {
            std::cerr << "encoder output did not contain image_embeddings\n";
            return 1;
        }
        if (!cvkit::infer::save_tensor_file(*embeddings, embeddings_path))
        {
            std::cerr << "failed to save embedding tensor: " << embeddings_path << '\n';
            return 1;
        }

        std::cout
            << "mode=" << promptable::mode_name(mode)
            << " image=" << image_path
            << " async=" << (async_infer ? "true" : "false")
            << " embeddings_output=" << embeddings_path
            << " embedding_shape=";
        for (std::size_t index = 0; index < embeddings->shape.size(); ++index)
        {
            if (index > 0)
            {
                std::cout << 'x';
            }
            std::cout << embeddings->shape[index];
        }
        std::cout << '\n';
        return 0;
    }

    const auto* mask_value = output.find<cvkit::infer::MaskValue>("mask");
    const auto* low_res_masks = output.find<cvkit::infer::TensorValue>("low_res_masks");
    const auto* logits = output.find<cvkit::infer::TensorValue>("logits");
    const auto* scores = output.find<std::vector<float>>("scores");
    if (mask_value == nullptr)
    {
        std::cerr << "segmentation output did not contain a mask\n";
        return 1;
    }

    const auto mask = cvkit::examples::frame_mask_to_mat(mask_value->frame);
    if (mask.empty())
    {
        std::cerr << "mask frame was empty\n";
        return 1;
    }

    cv::Mat overlay = image.clone();
    cv::Mat color_mask(image.rows, image.cols, CV_8UC3, cv::Scalar(64, 255, 64));
    color_mask.copyTo(overlay, mask);
    cv::addWeighted(overlay, 0.4, image, 0.6, 0.0, overlay);

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const auto output_root = std::filesystem::path(output_dir);
    const auto stem = std::filesystem::path(image_path).stem().string();
    const auto mask_path = output_root / (stem + "_sam_mask.png");
    const auto overlay_path = output_root / (stem + "_sam_overlay.png");
    const auto low_res_mask_path = output_root / (stem + "_sam_low_res_mask.png");
    const auto logits_path = output_root / (stem + "_sam_logits.png");
    const auto logits_text_path = output_root / (stem + "_sam_logits.txt");

    if (!cv::imwrite(mask_path.string(), mask))
    {
        std::cerr << "failed to write mask image: " << mask_path << '\n';
        return 1;
    }
    if (!cv::imwrite(overlay_path.string(), overlay))
    {
        std::cerr << "failed to write overlay image: " << overlay_path << '\n';
        return 1;
    }
    if (low_res_masks != nullptr)
    {
        const auto low_res_mask = promptable::first_low_res_mask_to_mat(*low_res_masks);
        if (!low_res_mask.empty() && !cv::imwrite(low_res_mask_path.string(), low_res_mask))
        {
            std::cerr << "failed to write low-res mask image: " << low_res_mask_path << '\n';
            return 1;
        }
    }
    if (logits != nullptr)
    {
        const auto logits_image = promptable::logits_to_mat(*logits);
        if (!logits_image.empty() && !cv::imwrite(logits_path.string(), logits_image))
        {
            std::cerr << "failed to write logits image: " << logits_path << '\n';
            return 1;
        }
        if (!promptable::write_tensor_text(*logits, logits_text_path))
        {
            return 1;
        }
    }

    std::cout
        << "mode=" << promptable::mode_name(mode)
        << ' '
        << "image=" << image_path
        << " async=" << (async_infer ? "true" : "false")
        << " embeddings_path=" << embeddings_path
        << " mask_output=" << mask_path
        << " overlay_output=" << overlay_path;
    if (low_res_masks != nullptr)
    {
        std::cout << " low_res_mask_output=" << low_res_mask_path;
    }
    if (logits != nullptr)
    {
        std::cout << " logits_output=" << logits_path
                  << " logits_text_output=" << logits_text_path;
    }
    if (scores != nullptr && !scores->empty())
    {
        std::cout << " best_score=" << (*scores)[0];
    }
    std::cout << '\n';
    return 0;
}
