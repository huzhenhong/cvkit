#include "cvkit/image/ops.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvkit::image
{

    cvkit::core::Frame resize(const cvkit::core::Frame& input, int width, int height)
    {
        cvkit::core::Frame output{};
        output.desc.width    = width;
        output.desc.height   = height;
        output.desc.channels = input.desc.channels;
        output.desc.format   = input.desc.format;
        output.pts           = input.pts;
        output.source        = input.source;

        if (input.desc.width <= 0 || input.desc.height <= 0 || input.desc.channels <= 0 || input.data.empty())
        {
            return output;
        }

        const auto input_type = input.desc.channels == 3 ? CV_8UC3 : CV_8UC1;
        cv::Mat    input_mat(input.desc.height, input.desc.width, input_type, const_cast<std::uint8_t*>(input.data.data()));
        cv::Mat    output_mat;
        cv::resize(input_mat, output_mat, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

        const auto bytes = output_mat.total() * output_mat.elemSize();
        output.data.assign(output_mat.data, output_mat.data + bytes);
        return output;
    }

}  // namespace cvkit::image
