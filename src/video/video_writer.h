/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-06-27 14:16:36
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-06-27 14:50:29
 * FilePath     : \\quickstart\\test\\util\\VideoWriter.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#ifdef __linux__
    #pragma once
    #include "hf_codecs_api.h"
    #include "opencv2/core/core.hpp"


namespace cvkit
{
    class VideoWriter
    {
      public:
        VideoWriter();

        ~VideoWriter();

        bool Open(const std::string&     savePath,
                  const shf_video_param& inVideoParam,
                  const shf_video_param& outVideoParam);

        void Close();

        bool Write(const cv::Mat& frame);

      private:
        HFVIDEO_ENCODER m_pVideoEncoder = nullptr;
    };
}  // namespace cvkit

#endif