#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/model.h"
#include "cvkit/infer/task_io.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace cvkit::examples::promptable
{

    enum class Mode : unsigned char
    {
        combined,
        encoder,
        decoder,
    };

    [[nodiscard]] cvkit::infer::CachePolicy parse_cache_policy(std::string_view value);
    [[nodiscard]] Mode                      parse_mode(std::string_view value);
    [[nodiscard]] std::string_view          mode_name(Mode mode);

    [[nodiscard]] bool                      write_tensor_text(
        const cvkit::infer::TensorValue& tensor,
        const std::filesystem::path&     path);
    [[nodiscard]] cv::Mat first_low_res_mask_to_mat(const cvkit::infer::TensorValue& tensor);
    [[nodiscard]] cv::Mat logits_to_mat(const cvkit::infer::TensorValue& tensor);

    [[nodiscard]] bool    dump_graph_json(
        const cvkit::infer::Model&   model,
        bool                         async_infer,
        Mode                         mode,
        std::string_view             embeddings_path,
        const std::filesystem::path& path);

    void print_graph_info(const cvkit::infer::Model& model);
    void print_graph_trace(const cvkit::infer::Model& model);

}  // namespace cvkit::examples::promptable
