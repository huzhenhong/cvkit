#include "image_preprocess.h"

#include "image_value_utils.h"
#include "opencv_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <utility>

namespace cvkit::infer::detail
{

    std::optional<cvkit::core::Frame> resolve_host_image_input(const TaskInput& input)
    {
        if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
        {
            return materialize_host_frame(*image);
        }
        if (const auto* frame = input.find<cvkit::core::Frame>("image"); frame != nullptr)
        {
            return *frame;
        }
        return std::nullopt;
    }

    std::vector<std::int64_t> resolve_nchw_input_shape(
        const IBackendSession&    backend,
        const cvkit::core::Frame& frame)
    {
        if (const auto* info = backend.input_info(0); info != nullptr && info->shape.size() == 4)
        {
            auto shape = info->shape;
            if (shape[0] <= 0)
            {
                shape[0] = 1;
            }
            if (shape[1] <= 0)
            {
                shape[1] = frame.desc.channels > 0 ? frame.desc.channels : 3;
            }
            if (shape[2] <= 0)
            {
                shape[2] = frame.desc.height;
            }
            if (shape[3] <= 0)
            {
                shape[3] = frame.desc.width;
            }
            return shape;
        }
        return {1, 3, frame.desc.height, frame.desc.width};
    }

    RawTensor build_rgb_nchw_float_input(
        const cvkit::core::Frame&        frame,
        const std::vector<std::int64_t>& input_shape,
        std::string                      tensor_name)
    {
        RawTensor tensor{};
        tensor.name  = std::move(tensor_name);
        tensor.shape = input_shape;

        if (input_shape.size() != 4 || input_shape[1] != 3 || input_shape[2] <= 0 || input_shape[3] <= 0)
        {
            return tensor;
        }

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
        cv::resize(
            rgb,
            resized,
            cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])),
            0.0,
            0.0,
            cv::INTER_LINEAR);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        const auto width      = static_cast<int>(input_shape[3]);
        const auto height     = static_cast<int>(input_shape[2]);
        const auto plane_size = static_cast<std::size_t>(width * height);
        tensor.data.resize(plane_size * 3U);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const auto pixel                       = resized.at<cv::Vec3f>(y, x);
                const auto offset                      = static_cast<std::size_t>(y * width + x);
                tensor.data[offset]                    = pixel[0];
                tensor.data[plane_size + offset]       = pixel[1];
                tensor.data[(2 * plane_size) + offset] = pixel[2];
            }
        }

        return tensor;
    }

}  // namespace cvkit::infer::detail
