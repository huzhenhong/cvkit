#pragma once
#include <curand_kernel.h>


namespace ieaa
{
    namespace cv_cuda
    {
        __global__ void DilateKenerl(const unsigned char* d_src,
                                     unsigned char*       d_dst,
                                     int                  img_width,
                                     int                  img_height,
                                     int                  kernel_x_offset_start,
                                     int                  kernel_x_offset_end,
                                     int                  kernel_y_offset_start,
                                     int                  kernel_y_offset_end);

        __global__ void ErodeKenerl(const unsigned char* d_src,
                                    unsigned char*       d_dst,
                                    int                  img_width,
                                    int                  img_height,
                                    int                  kernel_x_offset_start,
                                    int                  kernel_x_offset_end,
                                    int                  kernel_y_offset_start,
                                    int                  kernel_y_offset_end);

        __global__ void ResizeKernel(const unsigned char* d_src,
                                     unsigned char*       d_dst,
                                     int                  input_width,
                                     int                  input_height,
                                     int                  output_width,
                                     int                  output_height);

        __global__ void RgbToGrayKernel(const unsigned char* d_src,
                                        unsigned char*       d_dst,
                                        int                  width,
                                        int                  height);

    }  // namespace cv_cuda
}  // namespace ieaa
