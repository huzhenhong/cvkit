#pragma once
#include "../interface/ibackground_model.h"


namespace ieaa
{
    namespace background_model
    {
        class IViBe;
        class IPreprocess;
        class IPostprocess;

        class BackgroundModel : public IBackgroundModel
        {
          public:
            BackgroundModel(const unsigned char* img, const BackgroundModelParam& background_param);

            virtual ~BackgroundModel();

            void                 UpdateParam(const BackgroundModelParam& param) override;

            void                 Reset(const unsigned char* img) override;

            void                 Update(const unsigned char* img) override;

            void                 UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) override;

            const unsigned char* GetForeground() override;

          private:
            BackgroundModelParam          m_background_param;

            std::shared_ptr<IViBe>        m_vibe_sptr{nullptr};

            std::shared_ptr<IPreprocess>  m_pre_processor{nullptr};

            std::shared_ptr<IPostprocess> m_post_processor{nullptr};

            unsigned char*                m_pre_result{nullptr};

            unsigned char*                m_post_result{nullptr};

#ifdef BACKGROUND_MODEL_DEBUG
            size_t m_step{0};
#endif
        };

    }  // namespace background_model
}  // namespace ieaa