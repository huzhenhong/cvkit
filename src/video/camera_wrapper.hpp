/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-12-30 14:22:51
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-07-29 11:37:36
 * FilePath     : \\goodsmovedetector\\test\\cvkit\\video\\CameraWrapper.hpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/

#include "semaphore.hpp"
#include "video.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <regex>

#ifdef HAVE_OPENCV_HIGHGUI
    #include "opencv2/highgui/highgui.hpp"
#endif


namespace cvkit
{
    enum class eDataSourceType
    {
        NONE = 0,
        CAMERA,
        RTSP,
        VIDEO,
        IMG_FOLDER,
        IMG,
        TXT
    };


    template<typename ImgType>
    struct WrapperParam
    {
        std::string                                                  source;
        std::function<void(const ImgType& frame, uint64_t frameIdx)> frameCallBack;
        double                                                       extractInterval;
        bool                                                         bAsync;
        int                                                          cacheSize;
        bool                                                         bShow;
        std::string                                                  winname;
    };


    template<typename ImgType>
    class CameraWrapper
    {
      public:
        CameraWrapper(const WrapperParam<ImgType>& wrapperParam)
            : m_wrapperParam(wrapperParam)
        {
            m_frameCallBack = m_wrapperParam.frameCallBack;

            m_sourceType = JudgeSourceType(wrapperParam.source);
        }

        ~CameraWrapper(){};

        bool Execute()
        {
            auto ExitCallBack = [&]()
            {
                m_bExit = true;
                m_semaphore.Post();  // 唤醒主线程退出
            };

            auto frameCallBack = [&](const ImgType& frame, uint64_t frameIdx)
            {
                // auto pos = cv::getTrackbarPos("POS: ", m_wrapperParam.winname);
                // cv::setTrackbarPos("POS: ", m_wrapperParam.winname, pos);
                // fmt::print("pos: {}", pos);


                if (m_wrapperParam.bAsync)
                {
                    // 解码线程回调
                    while (m_frameQue.size() > m_wrapperParam.cacheSize)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }

                    {
                        std::lock_guard<std::mutex> lock(m_mtx);
                        m_frameQue.emplace(std::move(frame));
                    }

                    m_semaphore.Post();
                }
                else
                {
                    // 主线程
                    SyncRun(frame);
                }
            };

            SourceParam sourceParam;
            if (std::is_same<ImgType, cv::Mat>::value)
            {
                sourceParam.bCuda = false;
            }
            else
            {
                sourceParam.bCuda = true;
            }

            sourceParam.bAsync          = m_wrapperParam.bAsync;
            sourceParam.extractInterval = m_wrapperParam.extractInterval;
            sourceParam.source          = m_wrapperParam.source;
            sourceParam.exitCallBack    = ExitCallBack;

            if (eDataSourceType::CAMERA == m_sourceType || eDataSourceType::RTSP == m_sourceType)
            {
                m_cameraPtr = std::make_shared<Camera>(sourceParam, frameCallBack);
            }
            else if (eDataSourceType::VIDEO == m_sourceType)
            {
                m_cameraPtr = std::make_shared<Video>(sourceParam, frameCallBack);
            }

            if (m_wrapperParam.bAsync)
            {
                m_semaphore.SetThreshold(m_wrapperParam.cacheSize);
            }
            else
            {
                auto AtExitHandler = []()
                {
                    std::cout << "interrupt!\n";
                };

                std::atexit(AtExitHandler);
            }

#ifdef HAVE_OPENCV_HIGHGUI
            if (m_wrapperParam.bShow)
            {
                if (std::is_same<ImgType, cv::Mat>::value)
                {
                    cv::namedWindow(m_wrapperParam.winname, cv::WINDOW_NORMAL);
                }
                else
                {
                    cv::namedWindow(m_wrapperParam.winname, cv::WINDOW_OPENGL);
                }

                if (eDataSourceType::VIDEO == m_sourceType)
                {
                    auto PosCallBack = [](int pos, void* pUserData)
                    {
                        if (pUserData)
                        {
                            auto cap = (Video*)pUserData;
                            cap->SetPos(pos);
                        }
                    };

                    cv::createTrackbar(
                        "POS: ",
                        m_wrapperParam.winname,
                        0,
                        m_cameraPtr->GetTotalFrameNum(),
                        PosCallBack,
                        m_cameraPtr.get());
                }
            }
#endif

            if (!m_cameraPtr->Start())
            {
                return false;
            }

            // #ifdef HAVE_OPENCV_HIGHGUI
            //             cv::waitKey();
            // #endif
            if (m_wrapperParam.bAsync)
            {  // 主线程在这里阻塞
                AsyncRun();
            }

            m_cameraPtr->Stop();
            return true;
        }

      private:
        eDataSourceType JudgeSourceType(const std::string& source)
        {
            if (std::regex_match(source, std::regex{R"(^\d+$)"}))
            {
                std::cout << "is camera [" << source << "]" << std::endl;
                return eDataSourceType::CAMERA;
            }
            else if (std::regex_match(source, std::regex{R"(^rtsp://.+)"}))
            {
                std::cout << "is rtsp [" << source << "]" << std::endl;
                return eDataSourceType::CAMERA;
            }
            else if (std::regex_match(
                         source,
                         std::regex{
                             R"(([\w\/\-:]+[\w-]{1}\.(mp4|h264|avi|rmvb|mkv|asf|dav)$)|(([\w\/-:]+%\d+d\.(jpg|png|bmp))$))",
                             std::regex::icase}))
            {
                std::cout << "is video [" << source << "]" << std::endl;
                return eDataSourceType::VIDEO;
            }
            else if (std::regex_match(source, std::regex{R"([\w\/\-:]+)"}))
            {
                std::cout << "is image folder [" << source << "]" << std::endl;
                return eDataSourceType::IMG_FOLDER;
            }
            else if (std::regex_match(source, std::regex{R"([\w\/\-:]+[\w-]{1}\.(jpg|png|bmp)$)"}))
            {
                std::cout << "is image [" << source << "]" << std::endl;
                return eDataSourceType::IMG;
            }
            else if (std::regex_match(source, std::regex{R"([\w\/\-:]+[\w-]{1}\.txt$)"}))
            {
                std::cout << "is txt [" << source << "]" << std::endl;
                return eDataSourceType::TXT;
            }
            else
            {
                // return eDataSourceType::NONE;
                return eDataSourceType::VIDEO;
            }
        }


        void SyncRun(const ImgType& frame)
        {
#ifdef HAVE_OPENCV_HIGHGUI
            if (m_wrapperParam.bShow)
            {
                if (eDataSourceType::VIDEO == m_sourceType)
                {
                    cv::setTrackbarPos("POS: ", m_wrapperParam.winname, m_cameraPtr->GetPos());
                }

                cv::imshow(m_wrapperParam.winname, frame);

                if (27 == cv::waitKey(1))  // 回调帧率由上层控制
                {
                    std::exit(0);
                }
            }
#endif
            if (!frame.empty())
            {  // 图片传给上层
                m_frameCallBack(frame, uint64_t(m_cameraPtr->GetPos()));
            }
        }

        void AsyncRun()
        {
            while (true)
            {
                m_semaphore.Wait();  // 等待图片

                ImgType frame;

                if (!m_bExit)
                {
                    {
                        std::lock_guard<std::mutex> lock(m_mtx);
                        frame = m_frameQue.front();
                        m_frameQue.pop();
                    }
                }
                else  // 解码完成触发退出
                {
                    // 清空缓存
                    std::lock_guard<std::mutex> lock(m_mtx);
                    if (!m_frameQue.empty())
                    {
                        frame = m_frameQue.front();
                        m_frameQue.pop();
                    }
                    else
                    {
                        break;
                    }
                }

#ifdef HAVE_OPENCV_HIGHGUI
                if (m_wrapperParam.bShow)
                {
                    if (eDataSourceType::VIDEO == m_sourceType)
                    {
                        cv::setTrackbarPos("POS: ", m_wrapperParam.winname, m_cameraPtr->GetPos());
                    }

                    if (!frame.empty())
                    {
                        cv::imshow(m_wrapperParam.winname, frame);
                    }

                    if (27 == cv::waitKey(1))  // 回调帧率由上层控制
                    {
                        // 中途退出时队列中可能还有图片也不用管了
                        break;
                    }
                }
#endif

                if (!frame.empty())
                {  // 图片传给上层
                    m_frameCallBack(frame, uint64_t(m_cameraPtr->GetPos()));
                }
            }

#ifdef HAVE_OPENCV_HIGHGUI
            if (m_wrapperParam.bShow)
            {
                cv::destroyWindow(m_wrapperParam.winname);
            }
            std::cout << "frame: " << m_frameQue.size() << std::endl;
#endif
        }

      private:
        WrapperParam<ImgType>                                        m_wrapperParam;
        std::function<void(const ImgType& frame, uint64_t frameIdx)> m_frameCallBack;
        CameraPtr                                                    m_cameraPtr{nullptr};
        std::queue<ImgType>                                          m_frameQue;
        Semaphore                                                    m_semaphore;
        std::mutex                                                   m_mtx;
        std::atomic<bool>                                            m_bExit{false};
        eDataSourceType                                              m_sourceType{eDataSourceType::NONE};
    };

}  // namespace cvkit
