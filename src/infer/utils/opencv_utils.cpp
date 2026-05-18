#include "opencv_utils.h"

namespace cvkit::infer::detail
{

    cv::Mat frame_to_mat_copy(const cvkit::core::Frame& frame)
    {
        if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0 || frame.data.empty())
        {
            return {};
        }

        const auto type = frame.desc.channels == 3 ? CV_8UC3 : CV_8UC1;
        cv::Mat    mat(frame.desc.height, frame.desc.width, type, const_cast<std::uint8_t*>(frame.data.data()));
        return mat.clone();
    }

}  // namespace cvkit::infer::detail
