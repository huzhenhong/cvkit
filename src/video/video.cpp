/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-12-30 10:48:21
 * LastEditors  : huzhenhong
 * LastEditTime : 2021-12-30 10:58:02
 * FilePath     : \\GoodsMoveDetector\\src\\cvtoolkit\\implement\\Video.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "video.h"
#include <fmt/core.h>

namespace cvkit
{
    Video::Video(const SourceParam&                                           sourceParam,
                 std::function<void(const cv::Mat& frame, uint64_t frameIdx)> cpuCallBack)
        : Camera(sourceParam, cpuCallBack)
    {
        int i = 0;
        fmt::print("i: {}", i);
    };

#ifdef HAVE_OPENCV_CUDACODEC
    Video::Video(const SourceParam&                                                    sourceParam,
                 std::function<void(const cv::cuda::GpuMat& frame, uint64_t frameIdx)> cudaCallBack)
        : Camera(sourceParam, cudaCallBack){};
#endif

    Video::~Video(){};

    void Video::DecodeByCpu()
    {
        while (m_bRun)
        {
            if (m_video.set(cv::CAP_PROP_POS_FRAMES, m_pos))
            {
                ++m_pos;

                if (m_video.grab())
                {
                    if (0 == m_pos % m_extractFrameInterval)
                    {
                        cv::Mat frame;

                        if (m_video.retrieve(frame))
                        {
                            m_cpuCallBack(frame, uint64_t(m_pos.load()));
                        }
                    }
                }
            }

            if (m_pos >= m_totalFrameNum)
            {
                break;
            }
        }
    }

}  // namespace cvkit