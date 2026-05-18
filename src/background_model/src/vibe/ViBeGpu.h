
/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-19 02:02:12
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-08 06:45:21
 * FilePath     : \\goodsmovedetector\\src\\VibeGpu.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "IViBe.h"
#include <curand_kernel.h>


namespace ieaa
{
    namespace background_model
    {
        class VibeGpu : public IViBe
        {
          public:
            VibeGpu(const VibeParam& param, int width, int height);

            virtual ~VibeGpu();

            virtual void           UpdateParam(const VibeParam& param) override;

            virtual void           Reset(const unsigned char* img) override;

            virtual void           Update(const unsigned char* img, int threshold = INT_MAX) override;

            virtual void           UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) override;

            virtual unsigned char* GetForeground() override;

          private:
            VibeParam            m_param;

            int                  m_width;

            int                  m_height;

            unsigned char*       m_samples{nullptr};

            unsigned char*       m_foreground{nullptr};

            int*                 m_neighbour_x{nullptr};

            int*                 m_neighbour_y{nullptr};

            curandStateXORWOW_t* m_rand_state{nullptr};
        };

    }  // namespace background_model
}  // namespace ieaa