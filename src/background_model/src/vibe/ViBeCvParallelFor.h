
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
#include "ViBe.h"


namespace ieaa
{
    namespace background_model
    {

        class ViBeCvParallelFor : public ViBe
        {
          public:
            ViBeCvParallelFor(const VibeParam& param, int width, int height);

            virtual ~ViBeCvParallelFor();

            virtual void Update(const unsigned char* img, int foregroundT = INT_MAX) override;

            virtual void UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) override;
        };

    }  // namespace background_model
}  // namespace ieaa