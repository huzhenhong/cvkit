#pragma once

#include "cvkit/media/media_export.h"

#include "cvkit/core/types.h"
#include "cvkit/media/options.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace cvkit::media
{

    enum class BK_MEDIA_EXPORT WriterStatus : std::uint8_t
    {
        closed,
        open,
        limit_reached,
        invalid_options,
        unsupported_backend,
        backend_error,
    };

    struct BK_MEDIA_EXPORT WriterInfo
    {
        std::string   uri{};
        WriterBackend backend{WriterBackend::opencv};
        GstVideoCodec gst_codec{GstVideoCodec::jpegavi};
        bool          open{false};
        int           width{0};
        int           height{0};
        double        fps{0.0};
        std::size_t   frame_count{0};
        std::size_t   max_frames{0};
    };

    class BK_MEDIA_EXPORT Writer
    {
      public:
        Writer();
        ~Writer();

        Writer(Writer&&) noexcept;
        Writer& operator=(Writer&&) noexcept;

        Writer(const Writer&)            = delete;
        Writer& operator=(const Writer&) = delete;

        bool             open(WriterOptions options);
        bool             write(const cvkit::core::Frame& frame);
        bool             write(const cvkit::core::DeviceFrame& frame);
        void             close();
        bool             is_open() const;
        WriterStatus     status() const;
        std::string_view status_message() const;
        WriterInfo       info() const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::media
