#include "yolo_preprocess_cuda.h"

#if defined(CVKIT_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

#if defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    #include "yolo_preprocess_cuda_kernel.h"
#endif

namespace cvkit::infer::detail
{

    namespace
    {

#if defined(CVKIT_WITH_CUDA_RUNTIME)
        [[nodiscard]] bool cuda_ok(cudaError_t status)
        {
            return status == cudaSuccess;
        }
#endif

    }  // namespace

    std::optional<cvkit::core::Frame> copy_cuda_image_to_host_frame(
        const cvkit::infer::ImageValue& image,
        std::string*                    error_message)
    {
        if (!image.has_valid_device_view())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda image input does not expose a valid external device view";
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
        const auto packed_stride = image.packed_row_stride_bytes();
        const auto height = static_cast<std::size_t>(image.frame.desc.height);
        const auto bytes = packed_stride * height;
        if (bytes == 0U)
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda image input has an invalid packed layout";
            }
            return std::nullopt;
        }

        host_frame.data.assign(bytes, std::uint8_t{0});
        const auto source_stride = image.effective_row_stride_bytes();
        if (!cuda_ok(cudaMemcpy2D(
                host_frame.data.data(),
                packed_stride,
                image.external_data,
                source_stride,
                packed_stride,
                height,
                cudaMemcpyDeviceToHost)))
        {
            if (error_message != nullptr)
            {
                *error_message = "failed to copy cuda image input to host";
            }
            return std::nullopt;
        }

        return host_frame;
#endif
    }

    std::optional<LetterboxResult> preprocess_yolo_cuda(
        const cvkit::infer::ImageValue& image,
        const std::vector<std::int64_t>& input_shape,
        bool                             prefer_device_tensor_output,
        std::string*                     error_message)
    {
        if (!image.has_valid_device_view())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda image input does not expose a valid external device view";
            }
            return std::nullopt;
        }

#if defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
        LetterboxResult result{};
        if (preprocess_yolo_cuda_kernel(image, input_shape, prefer_device_tensor_output, result))
        {
            return result;
        }
        if (error_message != nullptr)
        {
            *error_message = "cuda kernel preprocess failed";
        }
        return std::nullopt;
#else
        auto host_frame = copy_cuda_image_to_host_frame(image, error_message);
        if (!host_frame.has_value())
        {
            return std::nullopt;
        }
        auto result = preprocess_yolo_cpu(*host_frame, input_shape);
        if (result.tensor.data.empty())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda_device preprocess failed after copying image to host";
            }
            return std::nullopt;
        }
        return result;
#endif
    }

}  // namespace cvkit::infer::detail
