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
                                               int                  random_range);

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
                                         int                  threshold);

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
                                            int                  y1);

    }  // namespace background_model
}  // namespace ieaa
