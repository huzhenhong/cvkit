/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-19 02:03:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-08 06:44:36
 * FilePath     : \\goodsmovedetector\\src\\VibeGpu.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "ViBeGpu.h"
#include "cudaVibeKernel.h"
#include "cv_cuda/interface/cudaCommon.h"


namespace ieaa
{
    namespace background_model
    {
        VibeGpu::VibeGpu(const VibeParam& param, int width, int height)
            : m_param(param)
            , m_width(width)
            , m_height(height)
        {
            if (3 == m_param.neighbourhood)
            {
                int neighbours_x[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
                int neighbours_y[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

                CUDA_CHECK(cudaMalloc(&m_neighbour_x, 9 * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&m_neighbour_y, 9 * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(m_neighbour_x, neighbours_x, 9 * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(m_neighbour_y, neighbours_y, 9 * sizeof(int), cudaMemcpyHostToDevice));
            }
            else
            {
                int neighbours_x[25] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2};
                int neighbours_y[25] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2};

                CUDA_CHECK(cudaMalloc(&m_neighbour_x, 25 * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&m_neighbour_y, 25 * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(m_neighbour_x, neighbours_x, 25 * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(m_neighbour_y, neighbours_y, 25 * sizeof(int), cudaMemcpyHostToDevice));
            }

            size_t img_size = m_width * m_height * 1 * sizeof(unsigned char);
            CUDA_CHECK(cudaMalloc(&m_foreground, img_size));
            CUDA_CHECK(cudaMalloc(&m_samples, img_size * (m_param.num_samples + 1) * sizeof(unsigned char)));
            CUDA_CHECK(cudaMalloc((void**)&m_rand_state, sizeof(curandStateXORWOW_t) * m_width * m_height));
        }

        VibeGpu::~VibeGpu()
        {
            CUDA_CHECK(cudaFree(m_neighbour_x));
            CUDA_CHECK(cudaFree(m_neighbour_y));
            CUDA_CHECK(cudaFree(m_foreground));
            CUDA_CHECK(cudaFree(m_samples));
            CUDA_CHECK(cudaFree(m_rand_state));
        }

        void VibeGpu::UpdateParam(const VibeParam& param)
        {
            m_param = param;
        }

        void VibeGpu::Reset(const unsigned char* img)
        {
            dim3 block_dim(32, 30);
            dim3 grid_dim((m_width + block_dim.x - 1) / block_dim.x,
                          (m_height + block_dim.y - 1) / block_dim.y);


            VibeResetSamplesKenerl<<<grid_dim, block_dim>>>(m_samples,
                                                            (unsigned char*)img,
                                                            m_foreground,
                                                            m_rand_state,
                                                            0,
                                                            m_neighbour_x,
                                                            m_neighbour_y,
                                                            m_width,
                                                            m_height,
                                                            m_param.neighbourhood,
                                                            m_param.num_samples,
                                                            m_param.min_matchs,
                                                            m_param.radius,
                                                            m_param.random_range);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void VibeGpu::Update(const unsigned char* img, int threshold)
        {
            dim3 block_dim(32, 30);
            dim3 grid_dim((m_width + block_dim.x - 1) / block_dim.x,
                          (m_height + block_dim.y - 1) / block_dim.y);

            VibeUpdateKenerl<<<grid_dim, block_dim>>>(m_samples,
                                                      (unsigned char*)img,
                                                      m_foreground,
                                                      m_rand_state,
                                                      m_neighbour_x,
                                                      m_neighbour_y,
                                                      m_width,
                                                      m_height,
                                                      m_param.neighbourhood,
                                                      m_param.num_samples,
                                                      m_param.min_matchs,
                                                      m_param.radius,
                                                      m_param.random_range,
                                                      threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void VibeGpu::UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1)
        {
            dim3 block_dim(32, 30);
            dim3 grid_dim((m_width + block_dim.x - 1) / block_dim.x,
                          (m_height + block_dim.y - 1) / block_dim.y);

            VibeUpdateRoiKenerl<<<grid_dim, block_dim>>>(m_samples,
                                                         (unsigned char*)img,
                                                         m_foreground,
                                                         m_rand_state,
                                                         m_neighbour_x,
                                                         m_neighbour_y,
                                                         m_width,
                                                         m_height,
                                                         m_param.neighbourhood,
                                                         m_param.num_samples,
                                                         m_param.min_matchs,
                                                         m_param.radius,
                                                         m_param.random_range,
                                                         x0,
                                                         y0,
                                                         x1,
                                                         y1);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        unsigned char* VibeGpu::GetForeground()
        {
            return m_foreground;
        }

    }  // namespace background_model
}  // namespace ieaa