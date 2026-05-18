#include "../interface/cudaKenerlWrapper.h"
#include "../interface/cudaCommon.h"
#include "cuda_kenerl_wrapper.h"


namespace ieaa
{
    namespace cv_cuda
    {
        using MorphologyKenelOp = void (*)(const unsigned char* d_src,
                                           unsigned char*       d_dst,
                                           int                  img_width,
                                           int                  img_height,
                                           int                  kernel_x_offset_start,
                                           int                  kernel_x_offset_end,
                                           int                  kernel_y_offset_start,
                                           int                  kernel_y_offset_end);

        void MorphologyOp(MorphologyKenelOp    op,
                          const unsigned char* d_src,
                          unsigned char*       d_dst,
                          unsigned char*       d_workspace,
                          int                  img_width,
                          int                  img_height,
                          int                  kernel_width,
                          int                  kernel_height,
                          int                  iterations)
        {
            static dim3    block_dim(32, 30);
            static dim3    grid_dim((img_width + block_dim.x - 1) / block_dim.x,
                                    (img_height + block_dim.y - 1) / block_dim.y);

            int            kernel_x_offset_start = -1 * (kernel_width >> 1);
            int            kernel_x_offset_end   = kernel_width >> 1;
            int            kernel_y_offset_start = -1 * (kernel_height >> 1);
            int            kernel_y_offset_end   = kernel_height >> 1;

            unsigned char* out = (iterations % 2 == 1) ? d_dst : d_workspace;

            op<<<grid_dim, block_dim>>>(d_src,
                                        out,
                                        img_width,
                                        img_height,
                                        kernel_x_offset_start,
                                        kernel_x_offset_end,
                                        kernel_y_offset_start,
                                        kernel_y_offset_end);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (iterations > 1)
            {
                unsigned char* in = out;
                out               = (iterations % 2 == 0) ? d_dst : d_workspace;

                for (int i = 1; i < iterations; ++i)
                {
                    op<<<grid_dim, block_dim>>>(in,
                                                out,
                                                img_width,
                                                img_height,
                                                kernel_x_offset_start,
                                                kernel_x_offset_end,
                                                kernel_y_offset_start,
                                                kernel_y_offset_end);
                    CUDA_CHECK(cudaDeviceSynchronize());

                    // 注意这里并没有修改 d_src 和 d_dst
                    unsigned char* tmp = in;
                    in                 = out;
                    out                = tmp;
                }
            }
        }

        void Dilate(const unsigned char* d_src,
                    unsigned char*       d_dst,
                    unsigned char*       d_workspace,
                    int                  img_width,
                    int                  img_height,
                    int                  kernel_width,
                    int                  kernel_height,
                    int                  iterations)
        {
            MorphologyOp(DilateKenerl,
                         d_src,
                         d_dst,
                         d_workspace,
                         img_width,
                         img_height,
                         kernel_width,
                         kernel_height,
                         iterations);
        }

        void Erode(const unsigned char* d_src,
                   unsigned char*       d_dst,
                   unsigned char*       d_workspace,
                   int                  img_width,
                   int                  img_height,
                   int                  kernel_width,
                   int                  kernel_height,
                   int                  iterations)
        {
            MorphologyOp(ErodeKenerl,
                         d_src,
                         d_dst,
                         d_workspace,
                         img_width,
                         img_height,
                         kernel_width,
                         kernel_height,
                         iterations);
        }


        void Morphology(const unsigned char*  d_src,
                        unsigned char*        d_dst,
                        unsigned char*        d_workspace,
                        unsigned char*        d_workspace_ex,
                        int                   img_width,
                        int                   img_height,
                        int                   kernel_width,
                        int                   kernel_height,
                        const MorphologyType& type,
                        int                   iterations)
        {
            switch (type)
            {
                case MorphologyType::OPEN:
                {
                    MorphologyOp(ErodeKenerl,
                                 d_src,
                                 d_workspace_ex,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);

                    MorphologyOp(DilateKenerl,
                                 d_workspace_ex,
                                 d_dst,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);
                    break;
                }

                case MorphologyType::CLOSE:
                {
                    MorphologyOp(DilateKenerl,
                                 d_src,
                                 d_workspace_ex,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);

                    MorphologyOp(ErodeKenerl,
                                 d_workspace_ex,
                                 d_dst,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);
                    break;
                }

                case MorphologyType::DILATE:
                {
                    MorphologyOp(DilateKenerl,
                                 d_src,
                                 d_dst,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);
                    break;
                }

                case MorphologyType::ERODE:
                {
                    MorphologyOp(ErodeKenerl,
                                 d_src,
                                 d_dst,
                                 d_workspace,
                                 img_width,
                                 img_height,
                                 kernel_width,
                                 kernel_height,
                                 iterations);
                    break;
                }

                default:
                {
                    printf("ERROR: %s:%d, unknown", __FILE__, __LINE__);
                    exit(1);
                    break;
                }
            }
        }

        void Resize(const unsigned char* d_input,
                    unsigned char*       d_output,
                    int                  input_width,
                    int                  input_height,
                    int                  output_width,
                    int                  output_height)
        {
            static dim3 block_dim(32, 30);
            static dim3 grid_dim((input_width + block_dim.x - 1) / block_dim.x,
                                 (input_height + block_dim.y - 1) / block_dim.y);

            ResizeKernel<<<block_dim, grid_dim>>>(d_input,
                                                  d_output,
                                                  input_width,
                                                  input_height,
                                                  output_width,
                                                  output_height);

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void RgbToGray(const unsigned char* src,
                       unsigned char*       dst,
                       int                  width,
                       int                  height)
        {
            static dim3 block_dim(32, 30);
            static dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                                 (height + block_dim.y - 1) / block_dim.y);

            RgbToGrayKernel<<<grid_dim, block_dim>>>(src, dst, width, height);

            CUDA_CHECK(cudaDeviceSynchronize());
        }

    }  // namespace cv_cuda
}  // namespace ieaa
