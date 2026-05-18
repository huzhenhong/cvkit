
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
#include <climits>
#include "../../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {

        class IViBe
        {
          public:
            IViBe(){};

            virtual ~IViBe(){};

            virtual void           UpdateParam(const VibeParam& param) = 0;

            virtual void           Reset(const unsigned char* img) = 0;

            virtual void           Update(const unsigned char* img, int threshold = INT_MAX) = 0;

            virtual void           UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) = 0;

            virtual unsigned char* GetForeground() = 0;
        };

    }  // namespace background_model
}  // namespace ieaa
