#pragma once

#include "cvkit/media/media_export.h"

#include "cvkit/core/types.h"
#include "cvkit/media/options.h"

#include <memory>
#include <string>
#include <string_view>

namespace cvkit::media
{

    class BK_MEDIA_EXPORT Source
    {
      public:
        Source();
        ~Source();

        Source(Source&&) noexcept;
        Source& operator=(Source&&) noexcept;

        Source(const Source&)            = delete;
        Source& operator=(const Source&) = delete;

        bool    open(std::string uri);
        bool    open(SourceOptions options);
        bool    read(cvkit::core::Frame& frame);
        void    close();

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::media
