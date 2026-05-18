#pragma once
#include "ipre_process.h"
#include "../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {
        class Prepreocss : public IPreprocess
        {
          public:
            explicit Prepreocss(const PreprocessParam& param);

            ~Prepreocss();

            void Process(const unsigned char* src, unsigned char*& dst) override;

          private:
            PreprocessParam m_param;

            unsigned char*  m_cvtcolor{nullptr};

            size_t          m_step{0};
        };

    }  // namespace background_model
}  // namespace ieaa
