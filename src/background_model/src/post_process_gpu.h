#pragma once
#include "ipost_process.h"
#include "../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {
        class PostprocessGpu : public IPostprocess
        {
          public:
            explicit PostprocessGpu(const PostprocessParam& param);

            ~PostprocessGpu();

            void Process(const unsigned char* src, unsigned char*& dst) override;

          private:
            PostprocessParam m_param;

            unsigned char*   m_workspace{nullptr};

            unsigned char*   m_workspace_ex{nullptr};

            unsigned char*   m_open{nullptr};

            unsigned char*   m_close{nullptr};

            unsigned char*   m_result{nullptr};
        };

    }  // namespace background_model
}  // namespace ieaa
