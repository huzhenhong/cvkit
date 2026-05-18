#include "promptable_preprocess_cuda.h"

#include "../../utils/image_value_utils.h"
#include "../../utils/opencv_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    #include "promptable_preprocess_cuda_kernel.h"
#endif

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

        [[nodiscard]] RawTensor build_encoder_input_from_host_frame(const cvkit::core::Frame& frame)
        {
            RawTensor tensor{};
            tensor.name = "batched_images";
            tensor.shape = {1, 3, kEncoderImageSize, kEncoderImageSize};

            auto source = frame_to_mat_copy(frame);
            if (source.empty())
            {
                return tensor;
            }

            cv::Mat rgb;
            if (source.channels() == 3)
            {
                if (frame.desc.format == cvkit::core::PixelFormat::rgb8)
                {
                    rgb = source;
                }
                else
                {
                    cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
                }
            }
            else
            {
                cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
            }

            cv::Mat resized;
            cv::resize(rgb, resized, cv::Size(kEncoderImageSize, kEncoderImageSize), 0.0, 0.0, cv::INTER_LINEAR);
            resized.convertTo(resized, CV_32FC3);

            tensor.data.resize(static_cast<std::size_t>(3 * kEncoderImageSize * kEncoderImageSize));
            const auto plane_size = static_cast<std::size_t>(kEncoderImageSize * kEncoderImageSize);
            for (int y = 0; y < kEncoderImageSize; ++y)
            {
                for (int x = 0; x < kEncoderImageSize; ++x)
                {
                    const auto pixel = resized.at<cv::Vec3f>(y, x);
                    const auto offset = static_cast<std::size_t>(y * kEncoderImageSize + x);
                    tensor.data[offset] = (pixel[0] - kSamMeanR) / kSamStdR;
                    tensor.data[plane_size + offset] = (pixel[1] - kSamMeanG) / kSamStdG;
                    tensor.data[(2 * plane_size) + offset] = (pixel[2] - kSamMeanB) / kSamStdB;
                }
            }

            return tensor;
        }

    }  // namespace

    std::optional<RawTensor> preprocess_promptable_encoder_cuda(
        const cvkit::infer::ImageValue& image,
        bool                            prefer_device_tensor_output,
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

#if defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
        RawTensor tensor{};
        if (preprocess_promptable_encoder_cuda_kernel(image, prefer_device_tensor_output, tensor))
        {
            return tensor;
        }
        if (error_message != nullptr)
        {
            *error_message = "cuda kernel preprocess failed";
        }
        return std::nullopt;
#else
        auto host_frame = materialize_host_frame(image, error_message);
        if (!host_frame.has_value())
        {
            return std::nullopt;
        }

        auto tensor = build_encoder_input_from_host_frame(*host_frame);
        if (tensor.data.empty())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda encoder preprocess failed after copying image to host";
            }
            return std::nullopt;
        }
        return tensor;
#endif
    }

}  // namespace cvkit::infer::detail
