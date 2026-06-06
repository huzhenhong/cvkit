#include "promptable_preprocess_cuda_kernel.h"

#include "../../utils/cuda_image_sampling.cuh"

#include <cuda_runtime_api.h>

#include <cmath>

namespace cvkit::infer::detail
{

    namespace
    {

        constexpr int   kEncoderImageSize = 1024;
        constexpr float kSamMeanR         = 123.675F;
        constexpr float kSamMeanG         = 116.28F;
        constexpr float kSamMeanB         = 103.53F;
        constexpr float kSamStdR          = 58.395F;
        constexpr float kSamStdG          = 57.12F;
        constexpr float kSamStdB          = 57.375F;

        __device__ float sample_channel_linear(
            const unsigned char* src,
            std::size_t          src_stride,
            int                  src_width,
            int                  src_height,
            float                src_x,
            float                src_y,
            int                  channel)
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

            const float v00 = static_cast<float>(row0[static_cast<std::size_t>(x0) * 3U + static_cast<std::size_t>(channel)]);
            const float v01 = static_cast<float>(row0[static_cast<std::size_t>(x1) * 3U + static_cast<std::size_t>(channel)]);
            const float v10 = static_cast<float>(row1[static_cast<std::size_t>(x0) * 3U + static_cast<std::size_t>(channel)]);
            const float v11 = static_cast<float>(row1[static_cast<std::size_t>(x1) * 3U + static_cast<std::size_t>(channel)]);

            const float top = v00 + (v01 - v00) * wx;
            const float bottom = v10 + (v11 - v10) * wx;
            return top + (bottom - top) * wy;
        }

        __global__ void promptable_preprocess_kernel(
            const unsigned char* src,
            std::size_t          src_stride,
            int                  src_width,
            int                  src_height,
            bool                 bgr_input,
            bool                 nv12_input,
            float*               dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= kEncoderImageSize || y >= kEncoderImageSize)
            {
                return;
            }

            const float src_x =
                ((static_cast<float>(x) + 0.5F) * static_cast<float>(src_width) / static_cast<float>(kEncoderImageSize)) - 0.5F;
            const float src_y =
                ((static_cast<float>(y) + 0.5F) * static_cast<float>(src_height) / static_cast<float>(kEncoderImageSize)) - 0.5F;

            float r = 0.0F;
            float g = 0.0F;
            float b = 0.0F;
            if (nv12_input)
            {
                cvkit::infer::detail::cuda::sample_nv12_rgb_linear_u8_range(
                    src,
                    src_stride,
                    src_width,
                    src_height,
                    src_x,
                    src_y,
                    r,
                    g,
                    b);
            }
            else
            {
                const float c0 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 0);
                const float c1 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 1);
                const float c2 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 2);

                r = c0;
                g = c1;
                b = c2;
                if (bgr_input)
                {
                    b = c0;
                    g = c1;
                    r = c2;
                }
            }

            const auto plane_size = static_cast<std::size_t>(kEncoderImageSize) * static_cast<std::size_t>(kEncoderImageSize);
            const auto dst_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(kEncoderImageSize) +
                                   static_cast<std::size_t>(x);
            dst[0 * plane_size + dst_index] = (r - kSamMeanR) / kSamStdR;
            dst[1 * plane_size + dst_index] = (g - kSamMeanG) / kSamStdG;
            dst[2 * plane_size + dst_index] = (b - kSamMeanB) / kSamStdB;
        }

    }  // namespace

    bool preprocess_promptable_encoder_cuda_kernel(
        const cvkit::infer::ImageValue& image,
        bool                            prefer_device_tensor_output,
        RawTensor&                      tensor)
    {
        const bool nv12_input = image.frame.desc.format == cvkit::core::PixelFormat::nv12;
        if (!image.has_valid_device_view() || (!nv12_input && image.frame.desc.channels != 3))
        {
            return false;
        }

        const auto plane_size = static_cast<std::size_t>(kEncoderImageSize) * static_cast<std::size_t>(kEncoderImageSize);
        const auto total_values = plane_size * 3U;

        float* device_output = nullptr;
        if (cudaMalloc(&device_output, total_values * sizeof(float)) != cudaSuccess)
        {
            return false;
        }

        const dim3 block_dim(16, 16);
        const dim3 grid_dim(
            static_cast<unsigned int>((kEncoderImageSize + block_dim.x - 1) / block_dim.x),
            static_cast<unsigned int>((kEncoderImageSize + block_dim.y - 1) / block_dim.y));

        const bool bgr_input = image.frame.desc.format != cvkit::core::PixelFormat::rgb8;
        promptable_preprocess_kernel<<<grid_dim, block_dim>>>(
            static_cast<const unsigned char*>(image.external_data),
            image.effective_row_stride_bytes(),
            image.frame.desc.width,
            image.frame.desc.height,
            bgr_input,
            nv12_input,
            device_output);

        if (cudaGetLastError() != cudaSuccess)
        {
            cudaFree(device_output);
            return false;
        }

        tensor.name = "batched_images";
        tensor.shape = {1, 3, kEncoderImageSize, kEncoderImageSize};
        tensor.data_type = cvkit::infer::TensorDataType::float32;

        if (prefer_device_tensor_output)
        {
            tensor.memory_device = cvkit::infer::MemoryDevice::cuda;
            tensor.storage = cvkit::infer::StorageKind::owned;
            tensor.external_data = device_output;
            tensor.storage_bytes = total_values * sizeof(float);
            tensor.storage_owner = std::shared_ptr<void>(
                device_output,
                [](void* ptr)
                {
                    if (ptr != nullptr)
                    {
                        cudaFree(ptr);
                    }
                });
        }
        else
        {
            tensor.memory_device = cvkit::infer::MemoryDevice::host;
            tensor.data.assign(total_values, 0.0F);
            const auto copy_status =
                cudaMemcpy(tensor.data.data(), device_output, total_values * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(device_output);
            if (copy_status != cudaSuccess)
            {
                tensor.data.clear();
                tensor.shape.clear();
                return false;
            }
        }

        return true;
    }

}  // namespace cvkit::infer::detail
