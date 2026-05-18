#pragma once


namespace ieaa
{
    namespace background_model
    {

        class IPreprocess
        {
          public:
            explicit IPreprocess() {};

            virtual ~IPreprocess() {};

            virtual void Process(const unsigned char* src, unsigned char*& dst) = 0;
        };

    }  // namespace background_model
}  // namespace ieaa
