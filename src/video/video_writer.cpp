#ifdef __linux__
    #include "video_writer.h"
    // #include "logger/Logger.h"
    #include "hf_video_encode.h"
    #include "opencv2/imgproc/imgproc.hpp"


namespace cvkit
{
    VideoWriter::VideoWriter() {}

    VideoWriter::~VideoWriter() {}

    bool VideoWriter::Open(const std::string&     savePath,
                           const shf_video_param& inVideoParam,
                           const shf_video_param& outVideoParam)
    {
        if (inVideoParam.pixfmt != HF_PIXFMT_NV12)
        {
            return false;
        }

        int            ret     = 0;
        unsigned char* pErrMsg = nullptr;

        hf_pixfmt      bestEncodeFmt = HF_PIXFMT_NONE;
        if ((ret = hf_video_enc_optimal_input_pixfmt(&bestEncodeFmt)) != 0)
        {
            hf_parse_error(ret, &pErrMsg);
            // LOG_ERROR("hf_video_enc_optimal_input_pixfmt failed, pErrMsg: [" << pErrMsg << "].");
            return false;
        }

        if (bestEncodeFmt != HF_PIXFMT_YUV420P)
        {
            // LOG_ERROR("best encode fmt is: [" << bestEncodeFmt << "] but must be HF_PIXFMT_YUV420P.");
            return false;
        }

        shf_video_enc_cfg encodeCfg;
        memset(&encodeCfg, 0, sizeof(shf_video_enc_cfg));
        strncpy((char*)encodeCfg.logtag, "sample_vencode", sizeof(encodeCfg.logtag));
        strncpy((char*)encodeCfg.oformat, "flv", sizeof(encodeCfg.oformat));
        strncpy((char*)encodeCfg.ovideo_codec_name, "h264", sizeof(encodeCfg.ovideo_codec_name));
        encodeCfg.device_id       = 0;  // only x86 platform valid, another platform is invalid but do not report error
        encodeCfg.iimage_pixfmt   = bestEncodeFmt;
        encodeCfg.ifps            = inVideoParam.fps;
        encodeCfg.iimage_width    = inVideoParam.width;
        encodeCfg.iimage_height   = inVideoParam.height;
        encodeCfg.oauto_reconnect = 0;
        encodeCfg.ofps            = outVideoParam.fps;
        encodeCfg.ovideo_width    = outVideoParam.width;
        encodeCfg.ovideo_height   = outVideoParam.height;
        encodeCfg.ovideo_bitrates = 0;
        // encodeCfg.ovideo_bitrates = (2 * 1024 * 1024) * (outVideoParam.width * outVideoParam.height) / (1920.0 * 1080.0);
        encodeCfg.opixfmt         = bestEncodeFmt;

        if ((ret = hf_create_video_encoder(encodeCfg, &m_pVideoEncoder)) != 0)
        {
            hf_parse_error(ret, &pErrMsg);
            // LOG_ERROR("hf_create_video_encoder failed, pErrMsg: [" << pErrMsg << "].");
            return false;
        }

        if ((ret = hf_video_encoder_pusher_open(m_pVideoEncoder,
                                                (const unsigned char*)savePath.data())) != 0)
        {
            hf_parse_error(ret, &pErrMsg);
            // LOG_ERROR("hf_video_encoder_pusher_open failed, pErrMsg: [" << pErrMsg << "].");
            return false;
        }

        return true;
    }

    void VideoWriter::Close()
    {
        if (m_pVideoEncoder)
        {
            int             ret      = 0;
            unsigned char*  pErrMsg  = nullptr;
            senc_write_ret* pRetInfo = nullptr;
            int             retNum   = 0;
            ret                      = hf_write_image_data(m_pVideoEncoder, nullptr, 0, &pRetInfo, &retNum);

            if (ret != 0)
            {
                unsigned char* pErrMsg = nullptr;
                hf_parse_error(ret, &pErrMsg);
                // LOG_ERROR("hf_write_image_data write last frame failed, pErrMsg: [" << pErrMsg << "].");
            }

            if ((ret = hf_video_encoder_pusher_close(m_pVideoEncoder, nullptr)) != 0)
            {
                hf_parse_error(ret, &pErrMsg);
                // LOG_ERROR("hf_video_encoder_pusher_close failed, pErrMsg: [" << pErrMsg << "].");
            }
        }
    }

    bool VideoWriter::Write(const cv::Mat& frame)
    {
        cv::Mat yuv420;
        cv::cvtColor(frame, yuv420, cv::COLOR_BGR2YUV_I420);

        // int             size       = frame.cols * frame.rows * 1.5;
        int             size     = yuv420.cols * yuv420.rows;
        senc_write_ret* pRetInfo = nullptr;
        int             retNum   = 0;

        int             ret = hf_write_image_data(m_pVideoEncoder, yuv420.data, size, &pRetInfo, &retNum);
        if (ret != 0)
        {
            unsigned char* pErrMsg = nullptr;
            hf_parse_error(ret, &pErrMsg);
            // LOG_ERROR("hf_write_image_data failed, pErrMsg: [" << pErrMsg << "].");
            return false;
        }

        if (pRetInfo)
        {
            hf_freep((void**)&pRetInfo);
        }

        return true;
    }
}  // namespace cvkit
#endif