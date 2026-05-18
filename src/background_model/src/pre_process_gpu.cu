#include "pre_process_gpu.h"
#include "cv_cuda/interface/cudaKenerlWrapper.h"
#include "cv_cuda/interface/cudaCommon.h"

#ifdef BACKGROUND_MODEL_TIMEIT
    #include "third_party/tracy/tracy/Tracy.hpp"
#endif


namespace ieaa
{
    namespace background_model
    {
        PrepreocssGpu::PrepreocssGpu(const PreprocessParam& param)
            : m_param(param)
        {
            CUDA_CHECK(cudaMalloc(&m_src, m_param.in_height * m_param.in_width * 3 * sizeof(unsigned char)));
            CUDA_CHECK(cudaMalloc(&m_dst, m_param.in_height * m_param.in_width * sizeof(unsigned char)));
        }

        PrepreocssGpu::~PrepreocssGpu()
        {
            cudaFree(m_src);
            cudaFree(m_dst);
        }


        void PrepreocssGpu::Process(const unsigned char* src, unsigned char*& dst)
        {
#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("PrepreocssGpu::Process, RgbToGray");
#endif

            int img_size = m_param.in_width * m_param.in_height * 3;

            {
#ifdef BACKGROUND_MODEL_TIMEIT
                ZoneScopedN("PrepreocssGpu::Process, copy image from cpu to gpu");
#endif
                CUDA_CHECK(cudaMemcpy(m_src, src, img_size, cudaMemcpyHostToDevice));
            }

            cv_cuda::RgbToGray(m_src,
                               m_dst,
                               m_param.in_width,
                               m_param.in_height);
            dst = m_dst;
        }
    }  // namespace background_model
}  // namespace ieaa
