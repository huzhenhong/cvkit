#pragma once
#include <curand_kernel.h>


namespace ieaa
{
    namespace cv_cuda
    {
        enum class MorphologyType
        {
            OPEN,
            CLOSE,
            DILATE,
            ERODE
        };

        void Dilate(const unsigned char* d_src,
                    unsigned char*       d_dst,
                    unsigned char*       d_workspace,
                    int                  img_width,
                    int                  img_height,
                    int                  kernel_width,
                    int                  kernel_height,
                    int                  iterations = 1);

        void Erode(const unsigned char* d_src,
                   unsigned char*       d_dst,
                   unsigned char*       d_workspace,
                   int                  img_width,
                   int                  img_height,
                   int                  kernel_width,
                   int                  kernel_height,
                   int                  iterations = 1);

        void Morphology(const unsigned char*  d_src,
                        unsigned char*        d_dst,
                        unsigned char*        d_workspace,
                        unsigned char*        d_workspace_ex,
                        int                   img_width,
                        int                   img_height,
                        int                   kernel_width,
                        int                   kernel_height,
                        const MorphologyType& type,
                        int                   iterations = 1);

        void Resize(const unsigned char* d_input,
                    unsigned char*       d_output,
                    int                  input_width,
                    int                  input_height,
                    int                  output_width,
                    int                  output_height);

        void RgbToGray(const unsigned char* src,
                       unsigned char*       dst,
                       int                  width,
                       int                  height);

    }  // namespace cv_cuda
}  // namespace ieaa
