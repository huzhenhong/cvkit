#pragma once
#include <memory>


namespace ieaa
{
    namespace background_model
    {

        struct VibeParam
        {
            int neighbourhood;
            int num_samples;
            int min_matchs;
            int radius;
            int random_range;
        };

        struct PreprocessParam
        {
            int in_width;
            int in_height;
            int out_width;
            int out_height;
        };

        struct PostprocessParam
        {
            int in_width;
            int in_height;
            int out_width;
            int out_height;
            int open_kernel;
            int close_kernel;
            int open_iterations;
            int close_iterations;
        };

        struct BackgroundModelParam
        {
            int              img_width;
            int              img_height;
            int              model_width;
            int              model_height;
            int              model_type;
            VibeParam        vibe_param;
            PreprocessParam  preprocess_param;
            PostprocessParam postprocess_param;
        };

        class IBackgroundModel
        {
          public:
            IBackgroundModel() = default;

            virtual ~IBackgroundModel() = default;

            virtual void                 UpdateParam(const BackgroundModelParam& param) = 0;

            virtual void                 Reset(const unsigned char* img) = 0;

            virtual void                 Update(const unsigned char* img) = 0;

            virtual void                 UpdateRoi(const unsigned char* img, int x0, int y0, int x1, int y1) = 0;

            virtual const unsigned char* GetForeground() = 0;
        };

        std::shared_ptr<IBackgroundModel> CreateBackgroundModel(const unsigned char*        img,
                                                                const BackgroundModelParam& cfg);

    }  // namespace background_model
}  // namespace ieaa