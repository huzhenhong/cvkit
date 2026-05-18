#include "post_process.h"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef BACKGROUND_MODEL_TIMEIT
    #include "third_party/tracy/tracy/Tracy.hpp"
#endif


#ifdef BACKGROUND_MODEL_DEBUG
    #include "opencv2/imgcodecs.hpp"
#endif


namespace ieaa
{
    namespace background_model
    {
        Postprocess::Postprocess(const PostprocessParam& param)
            : m_param(param)
        {
            m_open  = new unsigned char[m_param.in_height * m_param.in_width * sizeof(unsigned char)];
            m_close = new unsigned char[m_param.in_height * m_param.in_width * sizeof(unsigned char)];
        }

        Postprocess::~Postprocess()
        {
            delete[] m_open;
            delete[] m_close;
        }

        void Postprocess::Process(const unsigned char* src, unsigned char*& dst)
        {
#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("Postprocess::Process, morphology");
#endif
            cv::Mat input(m_param.in_height, m_param.in_width, CV_8UC1, (void*)src);

            cv::Mat opened(m_param.in_height, m_param.in_width, CV_8UC1, (void*)m_open);

            cv::Mat closed(m_param.in_height, m_param.in_width, CV_8UC1, (void*)m_close);

            auto    kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(m_param.open_kernel, m_param.open_kernel));
            cv::morphologyEx(input, opened, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), m_param.open_iterations);

#ifdef BACKGROUND_MODEL_DEBUG
            cv::imwrite("morphology_open.jpg", opened);
#endif
            auto kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(m_param.close_kernel, m_param.close_kernel));
            cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, kernel_close, cv::Point(-1, -1), m_param.close_iterations);
#ifdef BACKGROUND_MODEL_DEBUG
            cv::imwrite("morphology_close.jpg", closed);
#endif
            dst = m_close;
        }

    }  // namespace background_model
}  // namespace ieaa
