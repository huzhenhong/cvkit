#pragma once

#include "cvkit/infer/task_io.h"

#include <any>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cvkit::infer::detail
{

    struct NodeTrace
    {
        std::string   name{};
        std::size_t   sequence{0};
        std::size_t   input_count{0};
        std::size_t   output_count{0};
        std::size_t   scratch_count{0};
        std::uint64_t duration_us{0};
        bool          ok{true};
        std::string   message{};
    };

    struct Packet
    {
        TaskInput                                 input{};
        TaskOutput                                output{};
        std::unordered_map<std::string, std::any> scratch{};
        std::vector<NodeTrace>                    trace{};

        template<typename T>
        [[nodiscard]] T* get(std::string_view key)
        {
            const auto iter = scratch.find(std::string{key});
            if (iter == scratch.end())
            {
                return nullptr;
            }
            return std::any_cast<T>(&iter->second);
        }

        template<typename T>
        [[nodiscard]] const T* get(std::string_view key) const
        {
            const auto iter = scratch.find(std::string{key});
            if (iter == scratch.end())
            {
                return nullptr;
            }
            return std::any_cast<T>(&iter->second);
        }

        template<typename T>
        void put(std::string key, T&& value)
        {
            scratch.insert_or_assign(std::move(key), std::any{std::forward<T>(value)});
        }

        void add_trace(NodeTrace node_trace)
        {
            trace.push_back(std::move(node_trace));
        }
    };

}  // namespace cvkit::infer::detail
