#include "labels.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace cvkit::infer::detail
{
    namespace
    {

        [[nodiscard]] std::string trim_ascii(std::string value)
        {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
            {
                return {};
            }

            const auto last = value.find_last_not_of(" \t\r\n");
            return value.substr(first, last - first + 1);
        }

    }  // namespace

    bool load_labels_file(const std::string& path, std::vector<std::string>& labels)
    {
        labels.clear();
        if (path.empty() || !std::filesystem::exists(path))
        {
            return false;
        }

        std::ifstream input(path);
        if (!input.is_open())
        {
            return false;
        }

        for (std::string line; std::getline(input, line);)
        {
            auto trimmed = trim_ascii(std::move(line));
            if (trimmed.empty() || trimmed.front() == '#')
            {
                continue;
            }
            labels.push_back(std::move(trimmed));
        }

        return !labels.empty();
    }

    std::string resolve_label(const std::vector<std::string>& labels, int class_id)
    {
        if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels.size())
        {
            return labels[static_cast<std::size_t>(class_id)];
        }
        return "class_" + std::to_string(class_id);
    }

}  // namespace cvkit::infer::detail
