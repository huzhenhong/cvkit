#ifdef __linux__
    #pragma once
    #include "hf_codecs_api.h"
    #include "hf_video_decode.h"
    #include "opencv2/core/core.hpp"
    #include <string>
    #include <thread>
    #include <mutex>
    #include <queue>
    #include <atomic>
    #include <condition_variable>


namespace cvkit
{
    class VideoReader
    {
      public:
        VideoReader();
        ~VideoReader();

        bool                   Open(const std::string& videoPath);
        void                   Close();
        const shf_video_param& GetVideoParam();
        bool                   Read(cv::Mat& frame);

      private:
        void DecodeOneFrame();

      private:
        HFVIDEO_DECODER         m_pVideoDecoder{nullptr};
        shf_video_param         m_videoParam;
        std::queue<cv::Mat>     m_imgQue;
        std::mutex              m_mtx;
        std::condition_variable m_cdv;
        std::atomic<bool>       m_quit;
        int                     m_threshold{10};
        std::thread             m_decodeThread;
    };
}  // namespace cvkit

#endif