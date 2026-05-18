#pragma once
#include "ipost_process.h"
#include "../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {
        class Postprocess : public IPostprocess
        {
          public:
            explicit Postprocess(const PostprocessParam& param);

            ~Postprocess();

            void Process(const unsigned char* src, unsigned char*& dst) override;

          private:
            PostprocessParam m_param;

            unsigned char*   m_open{nullptr};

            unsigned char*   m_close{nullptr};
        };

    }  // namespace background_model
}  // namespace ieaa
