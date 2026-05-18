#include "example_opencv_utils.h"

namespace cvkit::examples
{

    cvkit::core::Frame mat_to_frame(const cv::Mat& image, std::string source)
    {
        cvkit::core::Frame frame{};
        frame.desc.width    = image.cols;
        frame.desc.height   = image.rows;
        frame.desc.channels = image.channels();
        frame.desc.format   = image.channels() == 3 ? cvkit::core::PixelFormat::bgr8 : cvkit::core::PixelFormat::unknown;
        frame.source        = std::move(source);

        if (!image.empty())
        {
            const auto bytes = image.total() * image.elemSize();
            frame.data.assign(image.data, image.data + bytes);
        }

        return frame;
    }

    cvkit::infer::ImageValue mat_to_image_value(const cv::Mat& image, std::string source)
    {
        cvkit::infer::ImageValue value{};
        value.frame = mat_to_frame(image, std::move(source));
        value.row_stride_bytes = image.empty() ? 0U : image.step[0];
        return value;
    }

    cv::Mat frame_mask_to_mat(const cvkit::core::Frame& mask)
    {
        if (mask.desc.width <= 0 || mask.desc.height <= 0 || mask.data.empty())
        {
            return {};
        }
        return cv::Mat(mask.desc.height, mask.desc.width, CV_8UC1, const_cast<std::uint8_t*>(mask.data.data())).clone();
    }

}  // namespace cvkit::examples
