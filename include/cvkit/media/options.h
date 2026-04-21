#pragma once

#include "cvkit/media/media_export.h"

#include <cstdint>
#include <string>

namespace cvkit::media
{

    enum class BK_MEDIA_EXPORT ReaderBackend : std::uint8_t
    {
        opencv,
        gstreamer,
        ffmpeg,
    };

    enum class BK_MEDIA_EXPORT WriterBackend : std::uint8_t
    {
        opencv,
        gstreamer,
        ffmpeg,
    };

    enum class BK_MEDIA_EXPORT GstVideoCodec : std::uint8_t
    {
        jpegavi,
        x264mp4,
        nvh264,
        nvv4l2h264,
    };

    struct BK_MEDIA_EXPORT SourceOptions
    {
        std::string   uri{};
        ReaderBackend backend{ReaderBackend::gstreamer};
    };

    struct BK_MEDIA_EXPORT WriterOptions
    {
        WriterBackend backend{WriterBackend::opencv};
        GstVideoCodec gst_codec{GstVideoCodec::jpegavi};
        std::size_t   max_frames{0};
    };

}  // namespace cvkit::media
