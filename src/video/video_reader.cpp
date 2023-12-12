#ifdef __linux__
    #include "video_reader.h"
    // #include "logger/Logger.h"
    #include <functional>
    #include "opencv2/imgproc/imgproc.hpp"


namespace cvkit
{
    VideoReader::VideoReader() {}

    VideoReader::~VideoReader()
    {
    }

    bool VideoReader::Open(const std::string& videoPath)
    {
        int            ret      = 0;
        unsigned char* errorMsg = nullptr;

        hf_pixfmt      pix_fmt;
        if ((ret = hf_video_dec_optimal_output_pixfmt(&pix_fmt)) != 0)
        {
            hf_parse_error(ret, &errorMsg);
            // LOG_ERROR("hf_video_dec_optimal_output_pixfmt failed, errorMsg: [" << errorMsg << "].");
            return false;
        }

        shf_video_dec_cfg decodeCfg;
        memset(&decodeCfg, 0, sizeof(shf_video_dec_cfg));  // set some default
        decodeCfg.out_pixfmt = pix_fmt;
        strncpy((char*)decodeCfg.logtag, "video decode", sizeof(decodeCfg.logtag));
        strncpy((char*)decodeCfg.stream_url, videoPath.data(), sizeof(decodeCfg.stream_url));


        if ((ret = hf_create_video_decoder(decodeCfg, &m_pVideoDecoder)) != 0)
        {
            hf_parse_error(ret, &errorMsg);
            // LOG_ERROR("hf_create_video_decoder failed, errorMsg: [" << errorMsg << "].");
            return false;
        }

        if ((ret = hf_video_stream_decoder_open(m_pVideoDecoder, &m_videoParam)) != 0)
        {
            hf_parse_error(ret, &errorMsg);
            // LOG_ERROR("hf_video_stream_decoder_open failed, errorMsg: [" << errorMsg << "].");
            return false;
        }

        m_quit.store(false);
        m_decodeThread = std::move(std::thread(std::bind(&VideoReader::DecodeOneFrame, this)));

        return true;
    }

    void VideoReader::Close()
    {
        m_quit.store(true);
        m_cdv.notify_all();

        if (m_decodeThread.joinable())
        {
            m_decodeThread.join();
        }

        if (m_pVideoDecoder)
        {
            int ret = hf_video_stream_decoder_close(m_pVideoDecoder);
            if (ret != 0)
            {
                unsigned char* errorMsg = nullptr;
                hf_parse_error(ret, &errorMsg);
                // LOG_ERROR("hf_video_stream_decoder_close failed, errorMsg: [" << errorMsg << "].");
            }
        }
    }

    const shf_video_param& VideoReader::GetVideoParam()
    {
        return m_videoParam;
    }

    bool VideoReader::Read(cv::Mat& frame)
    {
        std::unique_lock<std::mutex> ulk(m_mtx);  // 获得锁，准备消费

        if (m_imgQue.empty())
        {
            if (m_quit.load())
            {
                // 解码已完成
                return false;
            }

            m_cdv.wait(ulk);  // 没有存量，释放锁，等待生产
        }
        else
        {
            frame = m_imgQue.front();  // 正常消费
            m_imgQue.pop();
        }

        if (m_quit.load())
        {
            // 可能上层调用了Close
            return false;
        }

        if (m_imgQue.size() <= m_threshold)
        {
            m_cdv.notify_one();  // 存量过少，通知接着生产
        }

        return true;
    }

    void VideoReader::DecodeOneFrame()
    {
        int            size     = hf_get_decoded_data_size(m_videoParam);
        unsigned char* pBuf     = (unsigned char*)malloc(size);
        int            ret      = 0;
        unsigned char* errorMsg = nullptr;

        while (!m_quit.load())
        {
            svideo_dec_output outYuv;

            if ((ret = hf_get_decoded_data(m_pVideoDecoder, pBuf, size, &outYuv)) < 0)
            {
                hf_parse_error(ret, &errorMsg);
                // LOG_INFO("hf_get_decoded_data failed, errorMsg: [" << errorMsg << "].");
                break;  // 解码完成
            }

            cv::Mat tmp(outYuv.height * 3 / 2, outYuv.width, CV_8UC1, pBuf);
            cv::Mat img;
            cv::cvtColor(tmp, img, cv::COLOR_YUV2BGR_NV12);

            {
                std::unique_lock<std::mutex> ulk(m_mtx);  // 获得锁，准备生产
                while (!m_quit.load() && m_imgQue.size() > m_threshold)
                {
                    m_cdv.wait(ulk);  // 存量过多，释放锁，等待消费
                }
                m_imgQue.emplace(img);
                m_cdv.notify_one();  // 通知消费
            }
        }

        m_cdv.notify_one();  // 通知结束生产
        m_quit.store(true);

        if (pBuf)
        {
            free(pBuf);
            pBuf = nullptr;
        }
    }
}  // namespace cvkit

#endif