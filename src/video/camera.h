/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-12-30 10:42:16
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-05 11:01:29
 * FilePath     : \\GoodsMoveDetector\\src\\cvkit\\interface\\video\\Camera.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
// #include "opencv2/videoio.hpp"
#include "opencv2/videoio/videoio.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <thread>

#ifdef HAVE_OPENCV_CUDACODEC
    #include "opencv2/cudacodec.hpp"
#endif

namespace cvkit
{
    struct SourceParam
    {
        std::string           source;
        std::function<void()> exitCallBack;
        double                extractInterval;
        bool                  bAsync;
        bool                  bCuda;
    };


    class Camera
    {
      public:
        Camera(const SourceParam& sourceParam);

        Camera(const SourceParam&                                           sourceParam,
               std::function<void(const cv::Mat& frame, uint64_t frameIdx)> cpuCallBack);

#ifdef HAVE_OPENCV_CUDACODEC
        Camera(const SourceParam&                                                    sourceParam,
               std::function<void(const cv::cuda::GpuMat& frame, uint64_t frameIdx)> cudaCallBack);
#endif
        virtual ~Camera();

        bool   Start();
        void   Stop();
        double GetFps() const;
        int    GetTotalFrameNum() const;
        void   SetPos(int pos);
        int    GetPos() const;

      private:
        void         Run();
        virtual void DecodeByCuda();
        virtual void DecodeByCpu();

      protected:
        SourceParam                                                  m_sourceParam;
        cv::VideoCapture                                             m_video;
        std::function<void(const cv::Mat& frame, uint64_t frameIdx)> m_cpuCallBack{nullptr};

#ifdef HAVE_OPENCV_CUDACODEC
        cv::Ptr<cv::cudacodec::VideoReader>                                   m_videoCuda{nullptr};
        std::function<void(const cv::cuda::GpuMat& frame, uint64_t frameIdx)> m_cudaCallBack{nullptr};
#endif
        int               m_extractFrameInterval{5};
        double            m_fps{.0f};
        std::atomic<bool> m_bRun{true};
        std::thread       m_thread;
        int               m_totalFrameNum{0};
        std::atomic<int>  m_pos{0};
    };

    using CameraPtr = std::shared_ptr<Camera>;

}  // namespace cvkit