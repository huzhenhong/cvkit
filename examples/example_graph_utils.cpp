#include "example_graph_utils.h"

#include "cvkit/infer/debug.h"

#include <fstream>
#include <iostream>
#include <system_error>

namespace cvkit::examples
{

    bool dump_graph_json(
        const cvkit::infer::Model&   model,
        bool                         async_infer,
        const std::filesystem::path& path,
        std::string_view             extra_fields_json)
    {
        if (path.empty())
        {
            return true;
        }

        std::error_code ec;
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path(), ec);
        }

        std::ofstream output(path);
        if (!output.is_open())
        {
            std::cerr << "failed to open graph json output: " << path << '\n';
            return false;
        }

        output << cvkit::infer::build_graph_json(model, async_infer, std::string{extra_fields_json});
        return output.good();
    }

    void print_graph_info(std::ostream& stream, const cvkit::infer::Model& model)
    {
        cvkit::infer::print_graph_info(stream, model);
    }

    void print_graph_trace(std::ostream& stream, const cvkit::infer::Model& model)
    {
        cvkit::infer::print_graph_trace(stream, model);
    }

}  // namespace cvkit::examples
