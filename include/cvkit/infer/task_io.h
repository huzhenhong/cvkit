#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/infer_export.h"

#include <future>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cvkit::infer
{

    using Value = std::variant<
        cvkit::core::Frame,
        std::vector<cvkit::core::Detection>,
        std::vector<cvkit::core::Point2f>,
        std::string,
        std::vector<float>>;

    struct BK_INFER_EXPORT NamedValue
    {
        std::string name{};
        Value       value{};
    };

    struct BK_INFER_EXPORT TaskInput
    {
        std::vector<NamedValue> items{};

        template <typename T>
        [[nodiscard]] const T* find(std::string_view name) const
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                {
                    return std::get_if<T>(&item.value);
                }
            }
            return nullptr;
        }

        template <typename T>
        void add(std::string name, T&& value)
        {
            items.push_back(NamedValue{std::move(name), Value{std::forward<T>(value)}});
        }
    };

    struct BK_INFER_EXPORT TaskOutput
    {
        std::vector<NamedValue> items{};

        template <typename T>
        [[nodiscard]] const T* find(std::string_view name) const
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                {
                    return std::get_if<T>(&item.value);
                }
            }
            return nullptr;
        }

        template <typename T>
        void add(std::string name, T&& value)
        {
            items.push_back(NamedValue{std::move(name), Value{std::forward<T>(value)}});
        }
    };

    class BK_INFER_EXPORT TaskFuture
    {
      public:
        TaskFuture() = default;
        explicit TaskFuture(std::shared_future<TaskOutput> future)
            : future_(std::move(future))
        {
        }

        [[nodiscard]] bool valid() const
        {
            return future_.valid();
        }

        TaskOutput get()
        {
            return future_.get();
        }

      private:
        std::shared_future<TaskOutput> future_{};
    };

}  // namespace cvkit::infer
