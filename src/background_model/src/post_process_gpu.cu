#include "post_process_gpu.h"
#include "cv_cuda/interface/cudaKenerlWrapper.h"
#include "cv_cuda/interface/cudaCommon.h"

#ifdef BACKGROUND_MODEL_TIMEIT
    #include "third_party/tracy/tracy/Tracy.hpp"
#endif

#ifdef BACKGROUND_MODEL_DEBUG
    #include "opencv2/imgcodecs.hpp"
#endif


namespace ieaa
{
    namespace background_model
    {
        PostprocessGpu::PostprocessGpu(const PostprocessParam& param)
            : m_param(param)
        {
            size_t img_size = m_param.in_width * m_param.in_height * sizeof(unsigned char);

            CUDA_CHECK(cudaMalloc(&m_workspace, img_size));
            CUDA_CHECK(cudaMalloc(&m_workspace_ex, img_size));
            CUDA_CHECK(cudaMalloc(&m_open, img_size));
            CUDA_CHECK(cudaMalloc(&m_close, img_size));
            m_result = new unsigned char[m_param.in_height * m_param.in_width * sizeof(unsigned char)];

            cudaMemset(m_workspace, 0, img_size);
            cudaMemset(m_workspace_ex, 0, img_size);
        }

        PostprocessGpu::~PostprocessGpu()
        {
            cudaFree(m_workspace);
            cudaFree(m_workspace_ex);
            cudaFree(m_open);
            cudaFree(m_close);
            delete[] m_result;
        }


        void PostprocessGpu::Process(const unsigned char* src, unsigned char*& dst)
        {
#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("PostprocessGpu::Process, Morphology");
#endif
            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("PostprocessGpu::Process DILATE");
#endif
                cv_cuda::Morphology(src,
                                    m_open,
                                    m_workspace,
                                    m_workspace_ex,
                                    m_param.in_width,
                                    m_param.in_height,
                                    m_param.open_kernel,
                                    m_param.open_kernel,
                                    cv_cuda::MorphologyType::OPEN,
                                    m_param.open_iterations);
            }

#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat morphology_open;
            morphology_open.create(m_param.in_height, m_param.in_width, CV_8UC1);
            cudaMemcpy(morphology_open.data, m_open, m_param.in_width * m_param.in_height, cudaMemcpyDeviceToHost);
            cv::imwrite("morphology_open.jpg", morphology_open);
#endif
            cv_cuda::Morphology(m_open,
                                m_close,
                                m_workspace,
                                m_workspace_ex,
                                m_param.in_width,
                                m_param.in_height,
                                m_param.close_kernel,
                                m_param.close_kernel,
                                cv_cuda::MorphologyType::CLOSE,
                                m_param.close_iterations);

#ifdef BACKGROUND_MODEL_DEBUG
            cv::Mat morphology_close;
            morphology_close.create(m_param.in_height, m_param.in_width, CV_8UC1);
            cudaMemcpy(morphology_close.data, m_close, m_param.in_width * m_param.in_height, cudaMemcpyDeviceToHost);
            cv::imwrite("morphology_close.jpg", morphology_close);
#endif
            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("PostprocessGpu::Process, copy image from gpu to cpu");
#endif
                cudaMemcpy(m_result, m_close, m_param.in_width * m_param.in_height, cudaMemcpyDeviceToHost);
            }
            dst = m_result;
        }
    }  // namespace background_model
}  // namespace ieaa
