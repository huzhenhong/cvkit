#include "image_value_utils.h"

#if defined(CVKIT_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

namespace cvkit::infer::detail
{

    std::optional<cvkit::core::Frame> materialize_host_frame(
        const cvkit::infer::ImageValue& image,
        std::string*                    error_message)
    {
        if (image.memory_device == cvkit::infer::MemoryDevice::host)
        {
            if (!image.has_valid_host_layout())
            {
                if (error_message != nullptr)
                {
                    *error_message = "host image does not expose a valid host layout";
                }
                return std::nullopt;
            }
            return image.frame;
        }

        if (image.memory_device != cvkit::infer::MemoryDevice::cuda)
        {
            if (error_message != nullptr)
            {
                *error_message = "image memory device is not supported for host materialization";
            }
            return std::nullopt;
        }

        if (!image.has_valid_device_view())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda image does not expose a valid external device view";
            }
            return std::nullopt;
        }

#if !defined(CVKIT_WITH_CUDA_RUNTIME)
        if (error_message != nullptr)
        {
            *error_message = "cuda runtime is not available in this build";
        }
        return std::nullopt;
#else
        cvkit::core::Frame host_frame = image.frame;
        host_frame.data.resize(image.required_byte_size());

        const auto stride = image.effective_row_stride_bytes();
        if (image.frame.desc.format == cvkit::core::PixelFormat::nv12)
        {
            if (cudaMemcpy(
                    host_frame.data.data(),
                    image.external_data,
                    host_frame.data.size(),
                    cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                if (error_message != nullptr)
                {
                    *error_message = "failed to copy cuda nv12 image to host";
                }
                return std::nullopt;
            }
            return host_frame;
        }

        if (cudaMemcpy2D(
                host_frame.data.data(),
                stride,
                image.external_data,
                stride,
                static_cast<std::size_t>(image.frame.desc.width) * image.bytes_per_pixel(),
                image.frame.desc.height,
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            if (error_message != nullptr)
            {
                *error_message = "failed to copy cuda image to host";
            }
            return std::nullopt;
        }

        return host_frame;
#endif
    }

}  // namespace cvkit::infer::detail
