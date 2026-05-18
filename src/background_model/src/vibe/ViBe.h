
/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-08-19 02:02:12
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-09-08 06:45:21
 * FilePath     : \\goodsmovedetector\\src\\ViBe.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "IViBe.h"
#include <vector>


namespace ieaa
{
    namespace background_model
    {

        class ViBe : public IViBe
        {
          public:
            ViBe(const VibeParam& param, int width, int height);

            virtual ~ViBe();

            virtual void           UpdateParam(const VibeParam& param) override;

            virtual void           Reset(const unsigned char* img) override;

            virtual void           Update(const unsigned char* img, int threshold = INT_MAX) override;

            virtual void           UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) override;

            virtual unsigned char* GetForeground() override;

          protected:
            VibeParam        m_param;

            int              m_width{0};

            int              m_height{0};

            std::vector<int> m_neighbours_x;

            std::vector<int> m_neighbours_y;

            unsigned char*   m_foreground{nullptr};

            unsigned char*   m_samples{nullptr};
        };

    }  // namespace background_model
}  // namespace ieaa
