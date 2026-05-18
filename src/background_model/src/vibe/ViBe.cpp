/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-19 02:03:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-08 06:44:36
 * FilePath     : \\goodsmovedetector\\src\\ViBe.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "ViBe.h"
#include "opencv2/core.hpp"


namespace ieaa
{
    namespace background_model
    {

        ViBe::ViBe(const VibeParam& param, int width, int height)
            : m_param(param)
            , m_width(width)
            , m_height(height)
        {
            if (3 == m_param.neighbourhood)
            {
                m_neighbours_x.reserve(9);
                m_neighbours_y.reserve(9);
                m_neighbours_x.assign({-1, 0, 1, -1, 0, 1, -1, 0, 1});
                m_neighbours_y.assign({-1, 0, 1, -1, 0, 1, -1, 0, 1});
            }
            else
            {
                m_neighbours_x.reserve(25);
                m_neighbours_y.reserve(25);
                m_neighbours_x.assign({-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2});
                m_neighbours_y.assign({-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2});
            }

            m_samples    = new unsigned char[m_height * m_width * (m_param.num_samples + 1)];
            m_foreground = new unsigned char[m_width * m_height];
        }

        ViBe::~ViBe()
        {
            delete[] m_samples;
            delete[] m_foreground;
        }

        void ViBe::UpdateParam(const VibeParam& param)
        {
            m_param = param;
        }

        void ViBe::Reset(const unsigned char* img)
        {
            memset(m_foreground, 0, m_width * m_height);

            const int neighbourhood_squared = m_param.neighbourhood * m_param.neighbourhood;
            cv::RNG   rng(0);

            for (int r = 0; r < m_height; ++r)
            {
                for (int c = 0; c < m_width; ++c)
                {
                    for (int s = 0; s < m_param.num_samples; ++s)
                    {
                        // 随机选择 m_param.numSamples 个邻域像素点，构建背景模型
                        int rand_x = rng.uniform(0, neighbourhood_squared);
                        int rand_y = rng.uniform(0, neighbourhood_squared);
                        int row    = r + m_neighbours_y[rand_y];
                        int col    = c + m_neighbours_x[rand_x];

                        if (row < 0)
                        {
                            row = 0;
                        }
                        else if (row >= m_height)
                        {
                            row = m_height - 1;
                        }

                        if (col < 0)
                        {
                            col = 0;
                        }
                        else if (col >= m_width)
                        {
                            col = m_width - 1;
                        }

                        int pixel_offset            = (r * m_width + c) * (m_param.num_samples + 1);
                        m_samples[pixel_offset + s] = img[row * m_width + col];
                    }
                }
            }
        }

        void ViBe::Update(const unsigned char* img, int threshold)
        {
            cv::RNG rng(0);

            for (int r = 0; r < m_height; ++r)
            {
                for (int c = 0; c < m_width; ++c)
                {
                    int           pixel_offset = (r * m_width + c) * (m_param.num_samples + 1);
                    int           s            = 0;
                    int           matches      = 0;
                    unsigned char cur_pixel    = img[r * m_width + c];

                    for (; matches < m_param.min_matchs && s < m_param.num_samples; ++s)
                    {
                        if (abs(m_samples[pixel_offset + s] - cur_pixel) < m_param.radius)
                        {
                            ++matches;
                        }
                    }

                    if (matches >= m_param.min_matchs)  // 背景
                    {
                        m_samples[pixel_offset + m_param.num_samples] = 0;
                        m_foreground[r * m_width + c]                 = 0;

                        // 有 1 / φ 的概率去更新自己的模型样本值
                        if (0 == rng.uniform(0, m_param.random_range))
                        {
                            m_samples[pixel_offset + rng.uniform(0, m_param.num_samples)] = cur_pixel;
                        }

                        // // 同时也有 1 / φ 的概率去更新它的邻域点的模型样本值
                        // if (0 == rng.uniform(0, m_param.randomRange))
                        // {
                        //     int row = r + m_yNeighbourVec[rng.uniform(0, m_param.neighbourhood * m_param.neighbourhood)];
                        //     int col = c + m_xNeighbourVec[rng.uniform(0, m_param.neighbourhood * m_param.neighbourhood)];

                        //     // 防止越界
                        //     if (row < 0)
                        //     {
                        //         row = 0;
                        //     }
                        //     else if (row >= m_height)
                        //     {
                        //         row = m_height - 1;
                        //     }

                        //     if (col < 0)
                        //     {
                        //         col = 0;
                        //     }
                        //     else if (col >= m_width)
                        //     {
                        //         col = m_width - 1;
                        //     }

                        //     m_pSamples[row][col][rng.uniform(0, m_param.numSamples)] = cur_pixel;
                        // }
                    }
                    else  // 前景
                    {
                        ++(m_samples[pixel_offset + m_param.num_samples]);
                        m_foreground[r * m_width + c] = 255;

                        // 连续命中前景
                        if (m_samples[pixel_offset + m_param.num_samples] > threshold)
                        {
                            // 随机更新为背景
                            // m_pSamples[r][c][rng.uniform(0, m_param.numSamples)] = cur_pixel;
                            // m_foreground[r * m_width + c] = 0;  // 此时该点更新为背景

                            // 强制更新所有样本为背景
                            memset(&(m_samples[pixel_offset]), cur_pixel, m_param.num_samples);
                        }
                    }
                }
            }
        }

        void ViBe::UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1)
        {
            cv::RNG rng(0);

            for (int r = y0; r < y1; ++r)
            {
                for (int c = x0; c < x1; ++c)
                {
                    for (int s = 0; s < m_param.num_samples; ++s)
                    {
                        const int neighbourhood_squared = m_param.neighbourhood * m_param.neighbourhood;
                        int       row                   = r + m_neighbours_y[rng.uniform(0, neighbourhood_squared)];
                        int       col                   = c + m_neighbours_x[rng.uniform(0, neighbourhood_squared)];

                        if (row < 0)
                        {
                            row = 0;
                        }
                        else if (row >= m_height)
                        {
                            row = m_height - 1;
                        }

                        if (col < 0)
                        {
                            col = 0;
                        }
                        else if (col >= m_width)
                        {
                            col = m_width - 1;
                        }

                        int pixel_offset            = (r * m_width + c) * (m_param.num_samples + 1);
                        m_samples[pixel_offset + s] = img[row * m_width + col];
                    }
                }
            }
        }

        unsigned char* ViBe::GetForeground()
        {
            return m_foreground;
        }

    }  // namespace background_model
}  // namespace ieaa
