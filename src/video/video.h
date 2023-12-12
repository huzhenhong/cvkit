/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-12-30 10:47:34
 * LastEditors  : huzhenhong
 * LastEditTime : 2021-12-30 10:57:35
 * FilePath     : \\GoodsMoveDetector\\src\\cvtoolkit\\interface\\Video.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "camera.h"

namespace cvkit
{
    class Video : public Camera
    {
      public:
        Video(const SourceParam&                                           sourceParam,
              std::function<void(const cv::Mat& frame, uint64_t frameIdx)> cpuCallBack);

#ifdef HAVE_OPENCV_CUDACODEC
        Video(const SourceParam&                                                    sourceParam,
              std::function<void(const cv::cuda::GpuMat& frame, uint64_t frameIdx)> cudaCallBack);
#endif
        ~Video();

        virtual void DecodeByCpu() override;
    };

    using VideoPtr = std::shared_ptr<Video>;

}  // namespace cvkit