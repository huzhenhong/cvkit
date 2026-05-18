#pragma once

#include "cvkit/infer/model.h"

#include <filesystem>
#include <iosfwd>
#include <string_view>

namespace cvkit::examples
{

    [[nodiscard]] bool dump_graph_json(
        const cvkit::infer::Model&   model,
        bool                         async_infer,
        const std::filesystem::path& path,
        std::string_view             extra_fields_json = {});

    void print_graph_info(std::ostream& stream, const cvkit::infer::Model& model);
    void print_graph_trace(std::ostream& stream, const cvkit::infer::Model& model);

}  // namespace cvkit::examples
