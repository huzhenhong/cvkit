#pragma once

#include "cvkit/core/core_export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cvkit::core
{

    enum class BK_CORE_EXPORT PixelFormat : std::uint8_t
    {
        unknown,
        bgr8,
        rgb8,
    };

    struct BK_CORE_EXPORT ImageDesc
    {
        int         width{0};
        int         height{0};
        int         channels{0};
        PixelFormat format{PixelFormat::unknown};
    };

    struct BK_CORE_EXPORT Frame
    {
        ImageDesc            desc{};
        std::vector<uint8_t> data{};
        std::int64_t         pts{0};
        std::string          source{};
    };

    struct BK_CORE_EXPORT BBox
    {
        float x{0.0F};
        float y{0.0F};
        float width{0.0F};
        float height{0.0F};
    };

    struct BK_CORE_EXPORT Detection
    {
        BBox        box{};
        float       score{0.0F};
        int         class_id{-1};
        std::string label{};
    };

}  // namespace cvkit::core
