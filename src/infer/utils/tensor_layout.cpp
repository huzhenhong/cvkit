#include "tensor_layout.h"

#include <algorithm>
#include <vector>

namespace cvkit::infer::detail
{

    bool is_nchw_layout(const std::vector<std::int64_t>& shape)
    {
        return shape.size() == 4 && shape[1] > 0 && shape[1] <= 4;
    }

    std::vector<std::int64_t> resolve_input_shape(
        const std::vector<std::int64_t>& input_shape,
        const cvkit::core::Frame&        frame)
    {
        auto resolved = input_shape;
        if (resolved.empty())
        {
            return resolved;
        }

        const auto channels = std::max(1, frame.desc.channels);
        const auto height   = std::max(1, frame.desc.height);
        const auto width    = std::max(1, frame.desc.width);

        if (resolved.size() == 4)
        {
            if (resolved[0] <= 0)
            {
                resolved[0] = 1;
            }

            if (is_nchw_layout(resolved))
            {
                if (resolved[1] <= 0)
                {
                    resolved[1] = channels;
                }
                if (resolved[2] <= 0)
                {
                    resolved[2] = height;
                }
                if (resolved[3] <= 0)
                {
                    resolved[3] = width;
                }
            }
            else
            {
                if (resolved[1] <= 0)
                {
                    resolved[1] = height;
                }
                if (resolved[2] <= 0)
                {
                    resolved[2] = width;
                }
                if (resolved[3] <= 0)
                {
                    resolved[3] = channels;
                }
            }
        }

        return resolved;
    }

}  // namespace cvkit::infer::detail
