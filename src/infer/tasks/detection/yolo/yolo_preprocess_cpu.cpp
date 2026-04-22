#include "yolo_preprocess_cpu.h"

#include "../../../utils/tensor_layout.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cvkit::infer::detail
{
    namespace
    {

        [[nodiscard]] cv::Mat frame_to_mat(const cvkit::core::Frame& frame)
        {
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0 || frame.data.empty())
            {
                return {};
            }

            const auto type = frame.desc.channels == 3 ? CV_8UC3 : CV_8UC1;
            cv::Mat    mat(frame.desc.height, frame.desc.width, type, const_cast<std::uint8_t*>(frame.data.data()));
            return mat.clone();
        }

    }  // namespace

    LetterboxResult preprocess_yolo_cpu(
        const cvkit::core::Frame&        frame,
        const std::vector<std::int64_t>& input_shape)
    {
        LetterboxResult result{};
        if (!is_nchw_layout(input_shape))
        {
            return result;
        }

        const auto input_h = static_cast<int>(input_shape[2]);
        const auto input_w = static_cast<int>(input_shape[3]);
        if (input_h <= 0 || input_w <= 0)
        {
            return result;
        }

        auto source = frame_to_mat(frame);
        if (source.empty())
        {
            return result;
        }

        if (source.channels() == 3 && frame.desc.format != cvkit::core::PixelFormat::rgb8)
        {
            cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
        }

        const auto src_w     = static_cast<float>(source.cols);
        const auto src_h     = static_cast<float>(source.rows);
        const auto scale     = std::min(static_cast<float>(input_w) / src_w, static_cast<float>(input_h) / src_h);
        const auto resized_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
        const auto resized_h = std::max(1, static_cast<int>(std::round(src_h * scale)));

        cv::Mat resized;
        cv::resize(source, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat canvas(input_h, input_w, resized.type(), cv::Scalar(114, 114, 114));
        const auto pad_x = (input_w - resized_w) / 2;
        const auto pad_y = (input_h - resized_h) / 2;
        resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

        cv::Mat input_float;
        canvas.convertTo(input_float, CV_32F, 1.0 / 255.0);

        const auto channels   = static_cast<std::size_t>(input_float.channels());
        const auto plane_size = static_cast<std::size_t>(input_w * input_h);
        result.tensor.name    = "images";
        result.tensor.shape   = {1, static_cast<std::int64_t>(channels), input_h, input_w};
        result.tensor.data.assign(channels * plane_size, 0.0F);

        if (channels == 3)
        {
            std::vector<cv::Mat> split_channels;
            cv::split(input_float, split_channels);
            for (std::size_t c = 0; c < channels; ++c)
            {
                std::memcpy(
                    result.tensor.data.data() + c * plane_size,
                    split_channels[c].ptr<float>(),
                    plane_size * sizeof(float));
            }
        }
        else
        {
            std::memcpy(result.tensor.data.data(), input_float.ptr<float>(), plane_size * sizeof(float));
        }

        result.scale        = scale;
        result.pad_x        = static_cast<float>(pad_x);
        result.pad_y        = static_cast<float>(pad_y);
        result.input_width  = input_w;
        result.input_height = input_h;
        return result;
    }

}  // namespace cvkit::infer::detail
