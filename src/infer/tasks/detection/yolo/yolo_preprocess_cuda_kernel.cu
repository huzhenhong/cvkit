#include "yolo_preprocess_cuda_kernel.h"

#include "../../../utils/tensor_layout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>

namespace cvkit::infer::detail
{

    namespace
    {

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
            return (top + (bottom - top) * wy) / 255.0F;
        }

        __global__ void yolo_preprocess_kernel(
            const unsigned char* src,
            std::size_t          src_stride,
            int                  src_width,
            int                  src_height,
            bool                 bgr_input,
            float                scale,
            int                  resized_width,
            int                  resized_height,
            int                  pad_x,
            int                  pad_y,
            int                  dst_width,
            int                  dst_height,
            float*               dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= dst_width || y >= dst_height)
            {
                return;
            }

            const auto plane_size = static_cast<std::size_t>(dst_width) * static_cast<std::size_t>(dst_height);
            const auto dst_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) +
                                   static_cast<std::size_t>(x);
            const float fill_value = 114.0F / 255.0F;

            float r = fill_value;
            float g = fill_value;
            float b = fill_value;

            if (x >= pad_x && x < pad_x + resized_width && y >= pad_y && y < pad_y + resized_height)
            {
                const float local_x = static_cast<float>(x - pad_x);
                const float local_y = static_cast<float>(y - pad_y);
                const float src_x =
                    ((local_x + 0.5F) * static_cast<float>(src_width) / static_cast<float>(resized_width)) - 0.5F;
                const float src_y =
                    ((local_y + 0.5F) * static_cast<float>(src_height) / static_cast<float>(resized_height)) - 0.5F;
                const float c0 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 0);
                const float c1 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 1);
                const float c2 = sample_channel_linear(src, src_stride, src_width, src_height, src_x, src_y, 2);
                if (bgr_input)
                {
                    b = c0;
                    g = c1;
                    r = c2;
                }
                else
                {
                    r = c0;
                    g = c1;
                    b = c2;
                }
            }

            dst[0 * plane_size + dst_index] = r;
            dst[1 * plane_size + dst_index] = g;
            dst[2 * plane_size + dst_index] = b;
        }

    }  // namespace

    bool preprocess_yolo_cuda_kernel(
        const cvkit::infer::ImageValue& image,
        const std::vector<std::int64_t>& input_shape,
        bool                             prefer_device_tensor_output,
        LetterboxResult&                 result)
    {
        if (!image.has_valid_device_view() || !is_nchw_layout(input_shape))
        {
            return false;
        }
        if (image.frame.desc.channels != 3)
        {
            return false;
        }

        const auto input_h = static_cast<int>(input_shape[2]);
        const auto input_w = static_cast<int>(input_shape[3]);
        if (input_h <= 0 || input_w <= 0)
        {
            return false;
        }

        const auto src_w = static_cast<float>(image.frame.desc.width);
        const auto src_h = static_cast<float>(image.frame.desc.height);
        const auto scale = std::min(static_cast<float>(input_w) / src_w, static_cast<float>(input_h) / src_h);
        const auto resized_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
        const auto resized_h = std::max(1, static_cast<int>(std::round(src_h * scale)));
        const auto pad_x = (input_w - resized_w) / 2;
        const auto pad_y = (input_h - resized_h) / 2;

        const auto plane_size = static_cast<std::size_t>(input_w) * static_cast<std::size_t>(input_h);
        const auto total_values = plane_size * 3U;

        float* device_output = nullptr;
        if (cudaMalloc(&device_output, total_values * sizeof(float)) != cudaSuccess)
        {
            return false;
        }

        const dim3 block_dim(16, 16);
        const dim3 grid_dim(
            static_cast<unsigned int>((input_w + block_dim.x - 1) / block_dim.x),
            static_cast<unsigned int>((input_h + block_dim.y - 1) / block_dim.y));

        const bool bgr_input = image.frame.desc.format != cvkit::core::PixelFormat::rgb8;
        yolo_preprocess_kernel<<<grid_dim, block_dim>>>(
            static_cast<const unsigned char*>(image.external_data),
            image.effective_row_stride_bytes(),
            image.frame.desc.width,
            image.frame.desc.height,
            bgr_input,
            scale,
            resized_w,
            resized_h,
            pad_x,
            pad_y,
            input_w,
            input_h,
            device_output);

        const auto launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess)
        {
            cudaFree(device_output);
            return false;
        }

        result.tensor.name = "images";
        result.tensor.shape = {1, 3, input_h, input_w};
        result.tensor.data_type = cvkit::infer::TensorDataType::float32;

        if (prefer_device_tensor_output)
        {
            result.tensor.memory_device = cvkit::infer::MemoryDevice::cuda;
            result.tensor.storage = cvkit::infer::StorageKind::owned;
            result.tensor.external_data = device_output;
            result.tensor.storage_bytes = total_values * sizeof(float);
            result.tensor.storage_owner = std::shared_ptr<void>(
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
            result.tensor.memory_device = cvkit::infer::MemoryDevice::host;
            result.tensor.data.assign(total_values, 0.0F);
            const auto copy_status = cudaMemcpy(
                result.tensor.data.data(),
                device_output,
                total_values * sizeof(float),
                cudaMemcpyDeviceToHost);
            cudaFree(device_output);
            if (copy_status != cudaSuccess)
            {
                result.tensor.data.clear();
                result.tensor.shape.clear();
                return false;
            }
        }

        result.scale = scale;
        result.pad_x = static_cast<float>(pad_x);
        result.pad_y = static_cast<float>(pad_y);
        result.input_width = input_w;
        result.input_height = input_h;
        return true;
    }

}  // namespace cvkit::infer::detail
