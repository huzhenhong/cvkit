/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-12-30 10:42:16
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-05 11:04:37
 * FilePath     : \\GoodsMoveDetector\\src\\cvkit\\implement\\video\\Camera.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "camera.h"

namespace cvkit
{
    Camera::Camera(const SourceParam& sourceParam)
        : m_sourceParam(sourceParam)
    {
#ifdef HAVE_OPENCV_CUDACODEC
        if (sourceParam.bCuda)
        {
            auto cudaDevSize = cv::cuda::getCudaEnabledDeviceCount();
            for (int i = 0; i < cudaDevSize; ++i)
            {
                cv::cuda::printShortCudaDeviceInfo(i);
            }
            m_videoCuda = cv::cudacodec::createVideoReader(m_sourceParam.source);
        }
#endif
        // GPU 无法获取帧率、总帧数等信息，曲线救国吧
        m_video = cv::VideoCapture(m_sourceParam.source);
        if (m_video.isOpened())
        {
            // m_video.setExceptionMode(true);
            m_fps                  = m_video.get(cv::CAP_PROP_FPS);
            // m_extractFrameInterval = std::max(1, static_cast<int>(m_fps * m_sourceParam.extractInterval));
            m_extractFrameInterval = std::max<int>(1, static_cast<int>(1 * m_sourceParam.extractInterval));
            m_totalFrameNum        = static_cast<int>(m_video.get(cv::CAP_PROP_FRAME_COUNT));
            cv::Size size          = cv::Size(static_cast<int>(m_video.get(cv::CAP_PROP_FRAME_WIDTH)),
                                     static_cast<int>(m_video.get(cv::CAP_PROP_FRAME_HEIGHT)));

            std::cout << "source: " << m_sourceParam.source << std::endl;
            std::cout << "extract frame interval: " << m_extractFrameInterval << std::endl;
            std::cout << "total frame num: " << m_totalFrameNum << std::endl;
            std::cout << "fps: " << m_fps << std::endl;
            std::cout << "size: " << size << std::endl;
        }
        else
        {
            std::cout << "Open video [" << m_sourceParam.source << "] failed. check build imformatin below\n";
            std::cout << cv::getBuildInformation() << std::endl;
        }
        std::cout << cv::getBuildInformation() << std::endl;
    }

    Camera::Camera(const SourceParam&                                           sourceParam,
                   std::function<void(const cv::Mat& frame, uint64_t frameIdx)> cpuCallBack)
        : Camera(sourceParam)
    {
        m_cpuCallBack = cpuCallBack;
    };

#ifdef HAVE_OPENCV_CUDACODEC
    Camera::Camera(const SourceParam&                                                    sourceParam,
                   std::function<void(const cv::cuda::GpuMat& frame, uint64_t frameIdx)> cudaCallBack)
        : Camera(sourceParam)
    {
        m_cudaCallBack = cudaCallBack;
    };
#endif

    Camera::~Camera()
    {
        if (m_video.isOpened())
        {
            m_video.release();
        }

#ifdef HAVE_OPENCV_CUDACODEC
        if (m_sourceParam.bCuda && m_videoCuda)
        {
            m_videoCuda.release();
        }
#endif
    }


    bool Camera::Start()
    {
        if (!m_video.isOpened())
        {
            return false;
        }

        if (m_sourceParam.bCuda)
        {
#ifdef HAVE_OPENCV_CUDACODEC
            if (!m_cudaCallBack || !m_videoCuda)
            {
                return false;
            }
#endif
        }
        else
        {
            if (!m_cpuCallBack)
            {
                return false;
            }
        }

        m_bRun = true;

        if (m_sourceParam.bAsync)
        {
            m_thread = std::move(std::thread(std::bind(&Camera::Run, this)));
        }
        else
        {
            Run();
        }

        return true;
    }

    void Camera::Stop()
    {
        m_bRun = false;

        if (m_sourceParam.bAsync && m_thread.joinable())
        {
            m_thread.join();
        }
    }

    double Camera::GetFps() const
    {
        return m_fps;
    }

    int Camera::GetTotalFrameNum() const
    {
        return m_totalFrameNum;
    }

    void Camera::SetPos(int pos)
    {
        m_pos = pos;
    }

    int Camera::GetPos() const
    {
        return m_pos;
    }

    void Camera::Run()
    {
        if (m_sourceParam.bCuda)
        {
            DecodeByCuda();
        }
        else
        {
            DecodeByCpu();
        }

        if (m_sourceParam.exitCallBack)
        {
            m_sourceParam.exitCallBack();
        }
    }

    void Camera::DecodeByCuda()
    {
        int count{0};

        while (m_bRun)
        {
#ifdef HAVE_OPENCV_CUDACODEC
            cv::cuda::GpuMat frame;
            if (m_videoCuda->nextFrame(frame))
            {
                if (0 == count % m_extractFrameInterval)
                {
                    m_cudaCallBack(frame, count);
                }
            }
            ++count;
#else
            break;
#endif
        }
    }

    void Camera::DecodeByCpu()
    {
        int count{0};

        while (m_bRun)
        {
            if (m_video.grab())
            {
                if (0 == count % m_extractFrameInterval)
                {
                    cv::Mat frame;

                    if (m_video.retrieve(frame))
                    {
                        m_cpuCallBack(frame, count);
                    }
                }
            }

            ++count;
        }
    }

}  // namespace cvkit
