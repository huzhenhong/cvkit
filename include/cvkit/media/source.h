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

    enum class BK_MEDIA_EXPORT SourceStatus : std::uint8_t
    {
        closed,
        open,
        end_of_stream,
        invalid_uri,
        unsupported_backend,
        backend_error,
    };

    struct BK_MEDIA_EXPORT SourceInfo
    {
        std::string                 uri{};
        ReaderBackend               backend{ReaderBackend::opencv};
        bool                        open{false};
        int                         width{0};
        int                         height{0};
        int                         channels{0};
        cvkit::core::PixelFormat format{cvkit::core::PixelFormat::unknown};
        double                      fps{0.0};
        std::int64_t                frame_count{0};
        std::int64_t                frame_index{0};
    };

    struct BK_MEDIA_EXPORT RuntimeCapabilities
    {
        bool gstreamer{false};
        bool gstreamer_appsink{false};
        bool gstreamer_decodebin{false};
        bool gstreamer_h264parse{false};
        bool gstreamer_avdec_h264{false};
        bool gstreamer_nvh264dec{false};
        bool gstreamer_nvh264_device_decoder{false};
        bool gstreamer_cudaupload{false};
        bool gstreamer_cudadownload{false};
        bool gstreamer_cuda_convert{false};
    };

    BK_MEDIA_EXPORT RuntimeCapabilities runtime_capabilities(int cuda_device_index = 0);

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
        bool    read(cvkit::core::DeviceFrame& frame);
        void    close();
        bool    is_open() const;
        SourceStatus status() const;
        std::string_view status_message() const;
        SourceInfo info() const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::media
