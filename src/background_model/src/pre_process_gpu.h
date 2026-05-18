#pragma once
#include "ipre_process.h"
#include "../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {
        class PrepreocssGpu : public IPreprocess
        {
          public:
            explicit PrepreocssGpu(const PreprocessParam& param);

            ~PrepreocssGpu();

            void Process(const unsigned char* src, unsigned char*& dst) override;

          private:
            PreprocessParam m_param;

            unsigned char*  m_src{nullptr};

            unsigned char*  m_dst{nullptr};
        };

    }  // namespace background_model
}  // namespace ieaa
