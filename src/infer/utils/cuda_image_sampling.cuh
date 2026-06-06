#pragma once

#include <cstddef>

namespace cvkit::infer::detail::cuda
{

    __device__ inline float sample_plane_linear(
        const unsigned char* src,
        std::size_t          src_stride,
        int                  src_width,
        int                  src_height,
        float                src_x,
        float                src_y)
    {
        const float clamped_x = fminf(fmaxf(src_x, 0.0F), static_cast<float>(src_width - 1));
        const float clamped_y = fminf(fmaxf(src_y, 0.0F), static_cast<float>(src_height - 1));

        const int   x0 = static_cast<int>(floorf(clamped_x));
        const int   y0 = static_cast<int>(floorf(clamped_y));
        const int   x1 = min(x0 + 1, src_width - 1);
        const int   y1 = min(y0 + 1, src_height - 1);
        const float wx = clamped_x - static_cast<float>(x0);
        const float wy = clamped_y - static_cast<float>(y0);

        const auto* row0 = src + static_cast<std::size_t>(y0) * src_stride;
        const auto* row1 = src + static_cast<std::size_t>(y1) * src_stride;

        const float v00 = static_cast<float>(row0[x0]);
        const float v01 = static_cast<float>(row0[x1]);
        const float v10 = static_cast<float>(row1[x0]);
        const float v11 = static_cast<float>(row1[x1]);

        const float top    = v00 + (v01 - v00) * wx;
        const float bottom = v10 + (v11 - v10) * wx;
        return top + (bottom - top) * wy;
    }

    __device__ inline float sample_nv12_chroma_linear(
        const unsigned char* uv,
        std::size_t          uv_stride,
        int                  src_width,
        int                  src_height,
        float                src_x,
        float                src_y,
        int                  channel)
    {
        const int   chroma_width  = max(1, src_width / 2);
        const int   chroma_height = max(1, src_height / 2);
        const float chroma_x      = src_x * 0.5F;
        const float chroma_y      = src_y * 0.5F;
        const float clamped_x     = fminf(fmaxf(chroma_x, 0.0F), static_cast<float>(chroma_width - 1));
        const float clamped_y     = fminf(fmaxf(chroma_y, 0.0F), static_cast<float>(chroma_height - 1));

        const int   x0 = static_cast<int>(floorf(clamped_x));
        const int   y0 = static_cast<int>(floorf(clamped_y));
        const int   x1 = min(x0 + 1, chroma_width - 1);
        const int   y1 = min(y0 + 1, chroma_height - 1);
        const float wx = clamped_x - static_cast<float>(x0);
        const float wy = clamped_y - static_cast<float>(y0);

        const auto* row0 = uv + static_cast<std::size_t>(y0) * uv_stride;
        const auto* row1 = uv + static_cast<std::size_t>(y1) * uv_stride;

        const auto  offset00 = static_cast<std::size_t>(x0) * 2U + static_cast<std::size_t>(channel);
        const auto  offset01 = static_cast<std::size_t>(x1) * 2U + static_cast<std::size_t>(channel);
        const float v00      = static_cast<float>(row0[offset00]);
        const float v01      = static_cast<float>(row0[offset01]);
        const float v10      = static_cast<float>(row1[offset00]);
        const float v11      = static_cast<float>(row1[offset01]);

        const float top    = v00 + (v01 - v00) * wx;
        const float bottom = v10 + (v11 - v10) * wx;
        return top + (bottom - top) * wy;
    }

    __device__ inline void sample_nv12_rgb_linear_u8_range(
        const unsigned char* src,
        std::size_t          src_stride,
        int                  src_width,
        int                  src_height,
        float                src_x,
        float                src_y,
        float&               r,
        float&               g,
        float&               b)
    {
        const auto* uv = src + src_stride * static_cast<std::size_t>(src_height);
        const float y  = sample_plane_linear(src, src_stride, src_width, src_height, src_x, src_y);
        const float u  = sample_nv12_chroma_linear(uv, src_stride, src_width, src_height, src_x, src_y, 0);
        const float v  = sample_nv12_chroma_linear(uv, src_stride, src_width, src_height, src_x, src_y, 1);

        const float c     = y - 16.0F;
        const float d     = u - 128.0F;
        const float e     = v - 128.0F;
        const float raw_r = 1.164383F * c + 1.596027F * e;
        const float raw_g = 1.164383F * c - 0.391762F * d - 0.812968F * e;
        const float raw_b = 1.164383F * c + 2.017232F * d;
        r                 = fminf(fmaxf(raw_r, 0.0F), 255.0F);
        g                 = fminf(fmaxf(raw_g, 0.0F), 255.0F);
        b                 = fminf(fmaxf(raw_b, 0.0F), 255.0F);
    }

}  // namespace cvkit::infer::detail::cuda
