#include "pre_process.h"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef BACKGROUND_MODEL_TIMEIT
    #include "third_party/tracy/tracy/Tracy.hpp"
#endif


namespace ieaa
{
    namespace background_model
    {
        Prepreocss::Prepreocss(const PreprocessParam& param)
            : m_param(param)
        {
            m_cvtcolor = new unsigned char[m_param.in_height * m_param.in_width * sizeof(unsigned char)];
#ifdef _BUILD_SOPHON_
            m_step = cv::Mat(m_param.in_height, m_param.in_width, CV_8UC3).step;
#endif
        }

        Prepreocss::~Prepreocss()
        {
            delete[] m_cvtcolor;
        }

        void Prepreocss::Process(const unsigned char* src, unsigned char*& dst)
        {
#ifdef BACKGROUND_MODEL_TIMEIT
            ZoneScopedN("Prepreocss::Process, RgbToGray");
#endif
            cv::Mat input(m_param.in_height, m_param.in_width, CV_8UC3, (void*)src, m_step);

            cv::Mat output(m_param.in_height, m_param.in_width, CV_8UC1, (void*)m_cvtcolor);

            cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
            dst = m_cvtcolor;
        }

    }  // namespace background_model
}  // namespace ieaa
