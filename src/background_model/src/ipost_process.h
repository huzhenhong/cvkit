#pragma once


namespace ieaa
{
    namespace background_model
    {

        class IPostprocess
        {
          public:
            explicit IPostprocess() {};

            virtual ~IPostprocess() {};

            virtual void Process(const unsigned char* src, unsigned char*& dst) = 0;
        };

    }  // namespace background_model
}  // namespace ieaa
