#pragma once

#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    [[nodiscard]] bool        load_labels_file(const std::string& path, std::vector<std::string>& labels);
    [[nodiscard]] std::string resolve_label(const std::vector<std::string>& labels, int class_id);

}  // namespace cvkit::infer::detail
