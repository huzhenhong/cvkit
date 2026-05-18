#pragma once

#include "cvkit/infer/model.h"

#include <string_view>

namespace cvkit::examples
{

    [[nodiscard]] inline cvkit::infer::CachePolicy parse_cache_policy(std::string_view value)
    {
        if (value == "disabled")
        {
            return cvkit::infer::CachePolicy::disabled;
        }
        if (value == "read-only")
        {
            return cvkit::infer::CachePolicy::read_only;
        }
        if (value == "rebuild")
        {
            return cvkit::infer::CachePolicy::rebuild;
        }
        return cvkit::infer::CachePolicy::default_policy;
    }

}  // namespace cvkit::examples
