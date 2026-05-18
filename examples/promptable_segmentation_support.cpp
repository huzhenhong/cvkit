#include "promptable_segmentation_support.h"

#include "example_graph_utils.h"
#include "example_infer_utils.h"
#include "example_opencv_utils.h"
#include "cvkit/infer/debug.h"

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>
#include <vector>

namespace cvkit::examples::promptable
{

    namespace
    {

        [[nodiscard]] cv::Mat normalize_tensor_plane_to_mat(
            const cvkit::infer::TensorValue& tensor,
            std::size_t                      offset,
            int                              height,
            int                              width)
        {
            if (width <= 0 || height <= 0)
            {
                return {};
            }

            const auto plane_size = static_cast<std::size_t>(width * height);
            if (offset + plane_size > tensor.data.size())
            {
                return {};
            }

            cv::Mat values(
                height,
                width,
                CV_32FC1,
                const_cast<float*>(tensor.data.data() + static_cast<std::ptrdiff_t>(offset)));
            cv::Mat normalized;
            cv::normalize(values, normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
            return normalized.clone();
        }

    }  // namespace

    cvkit::infer::CachePolicy parse_cache_policy(std::string_view value)
    {
        return cvkit::examples::parse_cache_policy(value);
    }

    Mode parse_mode(std::string_view value)
    {
        if (value == "encoder")
        {
            return Mode::encoder;
        }
        if (value == "decoder")
        {
            return Mode::decoder;
        }
        return Mode::combined;
    }

    std::string_view mode_name(Mode mode)
    {
        switch (mode)
        {
            case Mode::encoder:
                return "encoder";
            case Mode::decoder:
                return "decoder";
            case Mode::combined:
            default:
                return "combined";
        }
    }

    bool write_tensor_text(const cvkit::infer::TensorValue& tensor, const std::filesystem::path& path)
    {
        std::error_code ec;
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path(), ec);
        }

        std::ofstream output(path);
        if (!output.is_open())
        {
            std::cerr << "failed to open tensor text output: " << path << '\n';
            return false;
        }

        output << "name=" << tensor.name << '\n';
        output << "shape=";
        for (std::size_t index = 0; index < tensor.shape.size(); ++index)
        {
            if (index > 0)
            {
                output << 'x';
            }
            output << tensor.shape[index];
        }
        output << '\n';
        output << "storage=" << cvkit::infer::storage_kind_name(tensor.storage) << '\n';
        output << "values=";
        for (std::size_t index = 0; index < tensor.data.size(); ++index)
        {
            if (index > 0)
            {
                output << ',';
            }
            output << tensor.data[index];
        }
        output << '\n';
        return output.good();
    }

    cv::Mat first_low_res_mask_to_mat(const cvkit::infer::TensorValue& tensor)
    {
        if (tensor.shape.size() != 5 || tensor.data.empty())
        {
            return {};
        }

        const auto mask_count = static_cast<std::size_t>(tensor.shape[2]);
        const auto height = static_cast<int>(tensor.shape[3]);
        const auto width = static_cast<int>(tensor.shape[4]);
        if (mask_count == 0 || width <= 0 || height <= 0)
        {
            return {};
        }

        return normalize_tensor_plane_to_mat(tensor, 0, height, width);
    }

    cv::Mat logits_to_mat(const cvkit::infer::TensorValue& tensor)
    {
        if (tensor.shape.size() != 3 || tensor.data.empty())
        {
            return {};
        }

        const auto height = static_cast<int>(tensor.shape[1]);
        const auto width = static_cast<int>(tensor.shape[2]);
        if (width <= 0 || height <= 0)
        {
            return {};
        }

        return normalize_tensor_plane_to_mat(tensor, 0, height, width);
    }

    bool dump_graph_json(
        const cvkit::infer::Model&        model,
        bool                              async_infer,
        Mode                              mode,
        std::string_view                  embeddings_path,
        const std::filesystem::path& path)
    {
        std::ostringstream extra;
        extra << "  \"mode\": \"" << mode_name(mode) << "\",\n";
        extra << "  \"aux_model_path\": \"" << model.aux_model_path() << "\",\n";
        extra << "  \"embeddings_path\": \"" << embeddings_path << "\",\n";
        return cvkit::examples::dump_graph_json(model, async_infer, path, extra.str());
    }

    void print_graph_info(const cvkit::infer::Model& model)
    {
        cvkit::examples::print_graph_info(std::cout, model);
    }

    void print_graph_trace(const cvkit::infer::Model& model)
    {
        cvkit::examples::print_graph_trace(std::cout, model);
    }

}  // namespace cvkit::examples::promptable
