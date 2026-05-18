#include <curand_kernel.h>


namespace ieaa
{
    namespace background_model
    {

        __global__ void VibeResetSamplesKenerl(unsigned char*       d_samples,
                                               unsigned char*       d_gray,
                                               unsigned char*       d_foreground,
                                               curandStateXORWOW_t* d_state,
                                               int                  seed,
                                               int*                 d_neighbour_x,
                                               int*                 d_neighbour_y,
                                               int                  width,
                                               int                  height,
                                               int                  neighbourhood,
                                               int                  num_samples,
                                               int                  min_matchs,
                                               int                  radius,
                                               int                  random_range)
        {
            int index_x  = threadIdx.x + blockIdx.x * blockDim.x;
            int index_y  = threadIdx.y + blockIdx.y * blockDim.y;
            int stride_x = blockDim.x * gridDim.x;
            int stride_y = blockDim.y * gridDim.y;

            for (int y = index_y; y < height; y += stride_y)
            {
                for (int x = index_x; x < width; x += stride_x)
                {
                    int pixel_offset  = x + y * width;
                    int samples_start = pixel_offset * (num_samples + 1);
                    curand_init(seed, pixel_offset, 0, d_state + pixel_offset);  // 每个线程都有自己的

                    for (int i = 0; i < num_samples; ++i)
                    {
                        int rand_x = (int)((curand_uniform(d_state + pixel_offset) - 1e-6f) * float(neighbourhood * neighbourhood));
                        int rand_y = (int)((curand_uniform(d_state + pixel_offset) - 1e-6f) * float(neighbourhood * neighbourhood));
                        int col    = x + d_neighbour_x[rand_x];
                        int row    = y + d_neighbour_y[rand_y];

                        if (row < 0)
                        {
                            row = 0;
                        }
                        else if (row >= height)
                        {
                            row = height - 1;
                        }

                        if (col < 0)
                        {
                            col = 0;
                        }
                        else if (col >= width)
                        {
                            col = width - 1;
                        }

                        d_samples[samples_start + i] = d_gray[col + row * width];
                    }

                    d_samples[samples_start + num_samples] = 0;
                    d_foreground[pixel_offset]             = 0;
                }
            }
        }


        __global__ void VibeUpdateKenerl(unsigned char*       d_samples,
                                         unsigned char*       d_gray,
                                         unsigned char*       d_foreground,
                                         curandStateXORWOW_t* d_state,
                                         int*                 d_neighbour_x,
                                         int*                 d_neighbour_y,
                                         int                  width,
                                         int                  height,
                                         int                  neighbourhood,
                                         int                  num_samples,
                                         int                  min_matchs,
                                         int                  radius,
                                         int                  random_range,
                                         int                  threshold)
        {
            int index_x  = threadIdx.x + blockIdx.x * blockDim.x;
            int index_y  = threadIdx.y + blockIdx.y * blockDim.y;
            int stride_x = blockDim.x * gridDim.x;
            int stride_y = blockDim.y * gridDim.y;

            for (int y = index_y; y < height; y += stride_y)
            {
                for (int x = index_x; x < width; x += stride_x)
                {
                    int           pixel_offset  = x + y * width;
                    int           samples_start = pixel_offset * (num_samples + 1);
                    int           matches       = 0;
                    int           s             = 0;
                    unsigned char cur_pixel     = d_gray[pixel_offset];

                    for (; matches < min_matchs && s < num_samples; ++s)
                    {
                        if (abs(d_samples[samples_start + s] - cur_pixel) < radius)
                        {
                            ++matches;
                        }
                    }

                    if (matches >= min_matchs)  // 背景
                    {
                        d_samples[samples_start + num_samples] = 0;
                        d_foreground[pixel_offset]             = 0;

                        // 有 1 / φ 的概率去更新自己的模型样本值
                        int rand = (int)((curand_uniform(d_state + pixel_offset) - 1e-6f) * random_range);
                        if (0 == rand)
                        {
                            int sampleOffset                        = (int)((curand_uniform(d_state + pixel_offset) - 1e-6f) * num_samples);
                            d_samples[samples_start + sampleOffset] = cur_pixel;
                        }
                    }
                    else  // 前景
                    {
                        // ++(d_samples[samples_start + num_samples]);
                        d_foreground[pixel_offset] = 255;

                        // // 连续命中前景
                        // if (d_samples[samples_start + num_samples] > threshold)
                        // {
                        //     // 强制更新所有样本为背景
                        //     for (int s = 0; s < num_samples; ++s)
                        //     {
                        //         d_samples[samples_start + s] = cur_pixel;
                        //     }
                        // }
                    }
                }
            }
        }

        __global__ void VibeUpdateRoiKenerl(unsigned char*       d_samples,
                                            unsigned char*       d_gray,

                                            unsigned char*       d_foreground,
                                            curandStateXORWOW_t* d_state,
                                            int*                 d_neighbour_x,
                                            int*                 d_neighbour_y,
                                            int                  width,
                                            int                  height,
                                            int                  neighbourhood,
                                            int                  num_samples,
                                            int                  min_matchs,
                                            int                  radius,
                                            int                  random_range,
                                            int                  x0,
                                            int                  y0,
                                            int                  x1,
                                            int                  y1)
        {
            int       index_x  = threadIdx.x + blockIdx.x * blockDim.x;
            int       index_y  = threadIdx.y + blockIdx.y * blockDim.y;
            int       stride_x = blockDim.x * gridDim.x;
            int       stride_y = blockDim.y * gridDim.y;

            const int neighbourhood_squared = neighbourhood * neighbourhood;

            for (int y = index_y; y <= y1; y += stride_y)
            {
                for (int x = index_x; x <= x1; x += stride_x)
                {
                    if (x0 <= x && x <= x1 && y0 <= y && y <= y1)
                    {
                        int pixel_offset  = x + y * width;
                        int samples_start = pixel_offset * (num_samples + 1);

                        for (int i = 0; i < num_samples; ++i)
                        {
                            int rand = (int)((curand_uniform(d_state + pixel_offset) - 1e-6f) * neighbourhood_squared);

                            int row = y + d_neighbour_y[rand];
                            int col = x + d_neighbour_x[rand];

                            if (row < 0)
                            {
                                row = 0;
                            }
                            else if (row >= height)
                            {
                                row = height - 1;
                            }

                            if (col < 0)
                            {
                                col = 0;
                            }
                            else if (col >= width)
                            {
                                col = width - 1;
                            }

                            d_samples[samples_start + i] = d_gray[col + row * width];
                        }

                        d_samples[samples_start + num_samples] = 0;
                        d_foreground[pixel_offset]             = 0;
                    }
                }
            }
        }

    }  // namespace background_model
}  // namespace ieaa
