/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-19 02:03:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-08 06:44:36
 * FilePath     : \\goodsmovedetector\\src\\ViBeCvParallelFor.cpp
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "ViBeCvParallelFor.h"
#include "opencv2/core.hpp"


namespace ieaa
{
    namespace background_model
    {

        ViBeCvParallelFor::ViBeCvParallelFor(const VibeParam& param, int width, int height)
            : ViBe(param, width, height)
        {
        }

        ViBeCvParallelFor::~ViBeCvParallelFor()
        {
        }

        void ViBeCvParallelFor::Update(const unsigned char* img, int threshold)
        {
            cv::parallel_for_(cv::Range(0, m_height),
                              [&](const cv::Range& range)
                              {
                                  cv::RNG rng(0);

                                  for (int r = range.start; r < range.end; ++r)
                                  {
                                      for (int c = 0; c < m_width; ++c)
                                      {
                                          int pixel_offset = (r * m_width + c) * (m_param.num_samples + 1);

                                          int s         = 0;
                                          int matches   = 0;
                                          int cur_pixel = img[r * m_width + c];

                                          for (matches = 0, s = 0; matches < m_param.min_matchs && s < m_param.num_samples; ++s)
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

                                              //   //   同时也有 1 / φ 的概率去更新它的邻域点的模型样本值
                                              //   if (0 == rng.uniform(0, m_param.random_range))
                                              //   {
                                              //       const int neighbourhood_squared = m_param.neighbourhood * m_param.neighbourhood;
                                              //       int              row                   = r + m_neighbours_y[rng.uniform(0, neighbourhood_squared)];
                                              //       int              col                   = c + m_neighbours_x[rng.uniform(0, neighbourhood_squared)];

                                              //       // 防止越界
                                              //       if (row < 0)
                                              //       {
                                              //           row = 0;
                                              //       }
                                              //       else if (row >= m_height)
                                              //       {
                                              //           row = m_height - 1;
                                              //       }

                                              //       if (col < 0)
                                              //       {
                                              //           col = 0;
                                              //       }
                                              //       else if (col >= m_width)
                                              //       {
                                              //           col = m_width - 1;
                                              //       }

                                              //       m_samples[row][col][rng.uniform(0, m_param.num_samples)] = cur_pixel;
                                              //   }
                                          }
                                          else  // 前景
                                          {
                                              ++(m_samples[pixel_offset + m_param.num_samples]);
                                              m_foreground[r * m_width + c] = 255;

                                              // 连续命中前景
                                              if (m_samples[pixel_offset + m_param.num_samples] > threshold)
                                              {
                                                  // // 随机更新为背景
                                                  // m_pSamples[r][c][rng.uniform(0, m_param.numSamples)] = cur_pix;
                                                  // m_foreground.at<uchar>(r, c)                                        = 0;  // 此时该点更新为背景

                                                  // 强制更新所有样本为背景
                                                  memset(&(m_samples[pixel_offset]), cur_pixel, m_param.num_samples);
                                              }
                                          }
                                      }
                                  }
                              });
        }

        void ViBeCvParallelFor::UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1)
        {
            cv::parallel_for_(cv::Range(y0, y1),
                              [&](const cv::Range& range)
                              {
                                  cv::RNG rng(0);

                                  for (int r = range.start; r < range.end; ++r)
                                  {
                                      for (int c = x0; c < x1; ++c)
                                      {
                                          for (int s = 0; s < m_param.num_samples; ++s)
                                          {
                                              const int        neighbourhood_squared = m_param.neighbourhood * m_param.neighbourhood;
                                              int              row                   = r + m_neighbours_y[rng.uniform(0, neighbourhood_squared)];
                                              int              col                   = c + m_neighbours_x[rng.uniform(0, neighbourhood_squared)];

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
                              });
        }

    }  // namespace background_model
}  // namespace ieaa
