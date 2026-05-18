#include "background_model.h"
#include "post_process.h"
#include "pre_process.h"
#include "vibe/ViBe.h"
#include "vibe/ViBeCvParallelFor.h"

#ifdef _BUILD_CUDA_
    #include "post_process_gpu.h"
    #include "pre_process_gpu.h"
    #include "vibe/ViBeGpu.h"
#endif

#ifdef BACKGROUND_MODEL_DEBUG
    #include <opencv2/imgcodecs.hpp>
#endif

#ifdef BACKGROUND_MODEL_TIMEIT
    #include "third_party/tracy/tracy/Tracy.hpp"
#endif

namespace ieaa
{
    namespace background_model
    {
        std::shared_ptr<IBackgroundModel> CreateBackgroundModel(const unsigned char*        img,
                                                                const BackgroundModelParam& cfg)
        {
            return std::make_shared<BackgroundModel>(img, cfg);
        }

        BackgroundModel::BackgroundModel(const unsigned char*        img,
                                         const BackgroundModelParam& background_param)
            : m_background_param(background_param)
        {
#ifdef BACKGROUND_MODEL_DEBUG
    #ifdef _BUILD_SOPHON_
            m_step = cv::Mat(m_background_param.img_height, m_background_param.img_width, CV_8UC3).step;
    #endif
#endif

#ifdef _BUILD_CUDA_
            cudaSetDevice(0);
#endif

            switch (m_background_param.model_type)
            {
                case 1:
                {
                    m_pre_processor  = std::make_shared<Prepreocss>(m_background_param.preprocess_param);
                    m_post_processor = std::make_shared<Postprocess>(m_background_param.postprocess_param);
                    m_vibe_sptr      = std::make_shared<ViBeCvParallelFor>(m_background_param.vibe_param,
                                                                      m_background_param.img_width,
                                                                      m_background_param.img_height);
                    break;
                }
#ifdef _BUILD_CUDA_
                case 2:
                {
                    m_pre_processor = std::make_shared<PrepreocssGpu>(m_background_param.preprocess_param);
                    m_post_processor =
                        std::make_shared<PostprocessGpu>(m_background_param.postprocess_param);
                    m_vibe_sptr = std::make_shared<VibeGpu>(m_background_param.vibe_param,
                                                            m_background_param.img_width,
                                                            m_background_param.img_height);
                    break;
                }
#endif
                case 0:
                default:
                {
                    m_pre_processor  = std::make_shared<Prepreocss>(m_background_param.preprocess_param);
                    m_post_processor = std::make_shared<Postprocess>(m_background_param.postprocess_param);
                    m_vibe_sptr =
                        std::make_shared<ViBe>(m_background_param.vibe_param, m_background_param.img_width, m_background_param.img_height);
                    break;
                }
            }

            Reset(img);
        }

        BackgroundModel::~BackgroundModel()
        {
        }

        void BackgroundModel::UpdateParam(const BackgroundModelParam& param)
        {
            m_vibe_sptr->UpdateParam(param.vibe_param);
        }

        void BackgroundModel::Reset(const unsigned char* img)
        {
#ifdef _BUILD_CUDA_
            cudaSetDevice(0);
#endif

#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("BackgroundModel::Reset");
#endif
#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat reset_src(m_background_param.img_height, m_background_param.img_width, CV_8UC3, (unsigned char*)img, m_step);
            cv::imwrite("reset_src.jpg", reset_src);

#endif
            m_pre_processor->Process(img, m_pre_result);

#ifdef BACKGROUND_MODEL_DEBUG
            if (m_background_param.model_type == 2)
            {
                cv::Mat reset_pre_process;
                reset_pre_process.create(m_background_param.img_height, m_background_param.img_width, CV_8UC1);
                cudaMemcpy(reset_pre_process.data, m_pre_result, m_background_param.img_height * m_background_param.img_width, cudaMemcpyDeviceToHost);
                cv::imwrite("reset_pre_process.jpg", reset_pre_process);
            }
            else
            {
                cv::Mat reset_pre_process(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_pre_result);
                cv::imwrite("reset_pre_process.jpg", reset_pre_process);
            }
#endif
            m_vibe_sptr->Reset(m_pre_result);

#ifdef BACKGROUND_MODEL_DEBUG
            if (m_background_param.model_type == 2)
            {
                cv::Mat reset_updated;
                reset_updated.create(m_background_param.img_height, m_background_param.img_width, CV_8UC1);
                cudaMemcpy(reset_updated.data, m_vibe_sptr->GetForeground(), m_background_param.img_height * m_background_param.img_width, cudaMemcpyDeviceToHost);
                cv::imwrite("reset_updated.jpg", reset_updated);
            }
            else
            {
                cv::Mat reset_updated(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_vibe_sptr->GetForeground());
                cv::imwrite("reset_updated.jpg", reset_updated);
            }
#endif
            m_post_processor->Process(m_vibe_sptr->GetForeground(),
                                      m_post_result);  // 防止 GetForeground() 返回空指针

#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat reset_foreground(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_post_result);
            cv::imwrite("reset_foreground.jpg", reset_foreground);
#endif
        }

        void BackgroundModel::Update(const unsigned char* img)
        {
#ifdef _BUILD_CUDA_
            cudaSetDevice(0);
#endif

#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("BackgroundModel::Update");
#endif

#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat src(m_background_param.img_height, m_background_param.img_width, CV_8UC3, (unsigned char*)img, m_step);
            cv::imwrite("src.jpg", src);
#endif

            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("BackgroundModel::Pre_process");
#endif
                m_pre_processor->Process(img, m_pre_result);
            }

#ifdef BACKGROUND_MODEL_DEBUG
            if (m_background_param.model_type == 2)
            {
                cv::Mat pre_process;
                pre_process.create(m_background_param.img_height, m_background_param.img_width, CV_8UC1);
                cudaMemcpy(pre_process.data, m_pre_result, m_background_param.img_height * m_background_param.img_width, cudaMemcpyDeviceToHost);
                cv::imwrite("pre_process.jpg", pre_process);
            }
            else
            {
                cv::Mat pre_process(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_pre_result);
                cv::imwrite("pre_process.jpg", pre_process);
            }
#endif
            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("BackgroundModel::vibe_Update");
#endif
                m_vibe_sptr->Update(m_pre_result);
            }
#ifdef BACKGROUND_MODEL_DEBUG
            if (m_background_param.model_type == 2)
            {
                cv::Mat updated;
                updated.create(m_background_param.img_height, m_background_param.img_width, CV_8UC1);
                cudaMemcpy(updated.data, m_vibe_sptr->GetForeground(), m_background_param.img_height * m_background_param.img_width, cudaMemcpyDeviceToHost);
                cv::imwrite("updated.jpg", updated);
            }
            else
            {
                cv::Mat updated(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_vibe_sptr->GetForeground());
                cv::imwrite("updated.jpg", updated);
            }
#endif
            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("BackgroundModel::Post_process");
#endif
                m_post_processor->Process(m_vibe_sptr->GetForeground(), m_post_result);
            }
#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat foreground(m_background_param.img_height, m_background_param.img_width, CV_8UC1, m_post_result);
            cv::imwrite("foreground.jpg", foreground);
#endif
        }

        void BackgroundModel::UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1)
        {
#ifdef _BUILD_CUDA_
            cudaSetDevice(0);
#endif

#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("BackgroundModel::vibe_UpdateRoi");
#endif
            m_vibe_sptr->UpdateRoi(m_pre_result, x0, y0, x1, y1);
        }

        const unsigned char* BackgroundModel::GetForeground()
        {
            return m_post_result;
        }

    }  // namespace background_model
}  // namespace ieaa