#include "cuda_opencv_kenerl.h"

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
                                     int                  kernel_y_offset_end)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < img_width && y < img_height)
            {
                unsigned char new_value = d_src[y * img_width + x];
                for (int i = kernel_x_offset_start; i <= kernel_x_offset_end; ++i)
                {
                    int neighbor_x = max(0, min(x + i, img_width - 1));
                    for (int j = kernel_y_offset_start; j <= kernel_y_offset_end; ++j)
                    {
                        int neighbor_y = max(0, min(y + j, img_height - 1));
                        new_value      = max(new_value, d_src[neighbor_y * img_width + neighbor_x]);
                    }
                }
                d_dst[y * img_width + x] = new_value;
            }
        }

        __global__ void ErodeKenerl(const unsigned char* d_src,
                                    unsigned char*       d_dst,
                                    int                  img_width,
                                    int                  img_height,
                                    int                  kernel_x_offset_start,
                                    int                  kernel_x_offset_end,
                                    int                  kernel_y_offset_start,
                                    int                  kernel_y_offset_end)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < img_width && y < img_height)
            {
                unsigned char new_value = d_src[y * img_width + x];
                for (int i = kernel_x_offset_start; i <= kernel_x_offset_end; ++i)
                {
                    int neighbor_x = max(0, min(x + i, img_width - 1));
                    for (int j = kernel_y_offset_start; j <= kernel_y_offset_end; ++j)
                    {
                        int neighbor_y = max(0, min(y + j, img_height - 1));
                        new_value      = min(new_value, d_src[neighbor_y * img_width + neighbor_x]);
                    }
                }
                d_dst[y * img_width + x] = new_value;
            }
        }

        __device__ unsigned char BilinearInterpolation(const unsigned char* input,
                                                       int                  inputWidth,
                                                       int                  inputHeight,
                                                       float                src_x,
                                                       float                src_y,
                                                       int                  channel)
        {
            int   x0 = (int)src_x;
            int   y0 = (int)src_y;
            int   x1 = x0 + 1;
            int   y1 = y0 + 1;

            float tx = src_x - x0;
            float ty = src_y - y0;

            // Boundary check
            x0 = max(0, min(x0, inputWidth - 1));
            x1 = max(0, min(x1, inputWidth - 1));
            y0 = max(0, min(y0, inputHeight - 1));
            y1 = max(0, min(y1, inputHeight - 1));

            float value = (1 - tx) * (1 - ty) * input[(y0 * inputWidth + x0) * 3 + channel] +
                          tx * (1 - ty) * input[(y0 * inputWidth + x1) * 3 + channel] +
                          (1 - tx) * ty * input[(y1 * inputWidth + x0) * 3 + channel] +
                          tx * ty * input[(y1 * inputWidth + x1) * 3 + channel];

            return (unsigned char)value;
        }

        __global__ void ResizeKernel(const unsigned char* d_src,
                                     unsigned char*       d_dst,
                                     int                  input_width,
                                     int                  input_height,
                                     int                  output_width,
                                     int                  output_height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < output_width && y < output_height)
            {
                float scale_x = (float)input_width / (float)output_width;
                float scale_y = (float)input_height / (float)output_height;

                float src_x = (x + 0.5f) * scale_x - 0.5f;
                float src_y = (y + 0.5f) * scale_y - 0.5f;

                for (int c = 0; c < 3; ++c)
                {  // 3 channels: R, G, B
                    d_dst[(y * output_width + x) * 3 + c] =
                        BilinearInterpolation(d_src, input_width, input_height, src_x, src_y, c);
                }
            }
        }

        __global__ void RgbToGrayKernel(const unsigned char* d_src,
                                        unsigned char*       d_dst,
                                        int                  width,
                                        int                  height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height)
            {
                int           idx = y * width + x;
                unsigned char r   = d_src[3 * idx];
                unsigned char g   = d_src[3 * idx + 1];
                unsigned char b   = d_src[3 * idx + 2];
#define CV_DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))
                d_dst[idx] = CV_DESCALE(9798 * r + 19235 * g + 3735 * b, 15);
            }
        }

    }  // namespace cv_cuda
}  // namespace ieaa
