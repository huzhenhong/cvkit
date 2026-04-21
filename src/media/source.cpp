#include "cvkit/media/source.h"

#if defined(CVKIT_WITH_GSTREAMER)
    #include <gst/app/gstappsink.h>
    #include <gst/video/video.h>
    #include <gst/gst.h>
#endif

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace cvkit::media
{

#if defined(CVKIT_WITH_GSTREAMER)
    namespace
    {

        void ensure_gstreamer_initialized()
        {
            static const bool initialized = []()
            {
                gst_init(nullptr, nullptr);
                return true;
            }();
            static_cast<void>(initialized);
        }

        [[nodiscard]] std::string normalize_uri(const std::string& uri)
        {
            if (uri.find("://") != std::string::npos)
            {
                return uri;
            }

            const auto path     = std::filesystem::absolute(uri);
            GError*    error    = nullptr;
            char*      file_uri = gst_filename_to_uri(path.c_str(), &error);
            if (file_uri == nullptr)
            {
                if (error != nullptr)
                {
                    g_error_free(error);
                }
                return uri;
            }

            std::string normalized(file_uri);
            g_free(file_uri);
            return normalized;
        }

        [[nodiscard]] std::string escape_location(const std::filesystem::path& path)
        {
            auto        location = std::filesystem::absolute(path).string();
            std::size_t offset   = 0;
            while ((offset = location.find('\\', offset)) != std::string::npos)
            {
                location.replace(offset, 1, "\\\\");
                offset += 2;
            }
            offset = 0;
            while ((offset = location.find('"', offset)) != std::string::npos)
            {
                location.replace(offset, 1, "\\\"");
                offset += 2;
            }
            return location;
        }

        [[nodiscard]] std::string make_pipeline_description(const std::string& uri)
        {
            if (uri.find("://") == std::string::npos && std::filesystem::exists(uri))
            {
                return "filesrc location=\"" + escape_location(uri) + "\" ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=cvkit_sink sync=false";
            }

            return "uridecodebin uri=\"" + normalize_uri(uri) + "\" ! videoconvert ! video/x-raw,format=BGR ! appsink name=cvkit_sink sync=false";
        }

        [[nodiscard]] bool check_bus_for_error(GstElement* pipeline)
        {
            auto* bus = gst_element_get_bus(pipeline);
            if (bus == nullptr)
            {
                return false;
            }

            bool ok = true;
            while (auto* message = gst_bus_pop_filtered(
                       bus,
                       static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING)))
            {
                switch (GST_MESSAGE_TYPE(message))
                {
                    case GST_MESSAGE_ERROR:
                    {
                        GError* error = nullptr;
                        gchar*  debug = nullptr;
                        gst_message_parse_error(message, &error, &debug);
                        if (error != nullptr)
                        {
                            std::cerr << "gstreamer error: " << error->message << '\n';
                            g_error_free(error);
                        }
                        if (debug != nullptr)
                        {
                            g_free(debug);
                        }
                        ok = false;
                        break;
                    }
                    case GST_MESSAGE_WARNING:
                    {
                        GError* error = nullptr;
                        gchar*  debug = nullptr;
                        gst_message_parse_warning(message, &error, &debug);
                        if (error != nullptr)
                        {
                            std::cerr << "gstreamer warning: " << error->message << '\n';
                            g_error_free(error);
                        }
                        if (debug != nullptr)
                        {
                            g_free(debug);
                        }
                        break;
                    }
                    default:
                        break;
                }
                gst_message_unref(message);
            }

            gst_object_unref(bus);
            return ok;
        }

        [[nodiscard]] cvkit::core::PixelFormat pixel_format_from_caps(const GstVideoInfo& info)
        {
            switch (GST_VIDEO_INFO_FORMAT(&info))
            {
                case GST_VIDEO_FORMAT_BGR:
                    return cvkit::core::PixelFormat::bgr8;
                case GST_VIDEO_FORMAT_RGB:
                    return cvkit::core::PixelFormat::rgb8;
                default:
                    return cvkit::core::PixelFormat::unknown;
            }
        }

    }  // namespace
#endif

    class Source::Impl
    {
      public:
        ~Impl()
        {
            close();
        }

        bool open(SourceOptions options)
        {
            close();
            backend_ = options.backend;
            uri_     = std::move(options.uri);
            if (uri_.empty())
            {
                return false;
            }

#if defined(CVKIT_WITH_GSTREAMER)
            if (backend_ != ReaderBackend::gstreamer)
            {
                return false;
            }
            ensure_gstreamer_initialized();
            const auto pipeline_description = make_pipeline_description(uri_);

            GError*    error = nullptr;
            pipeline_        = gst_parse_launch(pipeline_description.c_str(), &error);
            if (pipeline_ == nullptr)
            {
                if (error != nullptr)
                {
                    g_error_free(error);
                }
                return false;
            }

            sink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "cvkit_sink"));
            if (sink_ == nullptr)
            {
                close();
                return false;
            }

            gst_app_sink_set_emit_signals(sink_, FALSE);
            gst_app_sink_set_drop(sink_, FALSE);
            gst_app_sink_set_max_buffers(sink_, 4);

            const auto state_change = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
            if (state_change == GST_STATE_CHANGE_FAILURE)
            {
                close();
                return false;
            }

            GstState current = GST_STATE_NULL;
            GstState pending = GST_STATE_NULL;
            gst_element_get_state(pipeline_, &current, &pending, 5 * GST_SECOND);

            if (!check_bus_for_error(pipeline_))
            {
                close();
                return false;
            }

            open_ = true;
            return true;
#else
            static_cast<void>(backend_);
            open_ = false;
            return false;
#endif
        }

        bool read(cvkit::core::Frame& frame)
        {
            if (!open_)
            {
                return false;
            }

#if defined(CVKIT_WITH_GSTREAMER)
            GstSample* sample = nullptr;
            if (!primed_)
            {
                sample  = gst_app_sink_try_pull_preroll(sink_, 15 * GST_SECOND);
                primed_ = true;
            }
            else
            {
                sample = gst_app_sink_try_pull_sample(sink_, 15 * GST_SECOND);
            }
            if (sample == nullptr)
            {
                static_cast<void>(check_bus_for_error(pipeline_));
                return false;
            }

            auto* caps   = gst_sample_get_caps(sample);
            auto* buffer = gst_sample_get_buffer(sample);
            if (caps == nullptr || buffer == nullptr)
            {
                gst_sample_unref(sample);
                return false;
            }

            GstVideoInfo info{};
            if (!gst_video_info_from_caps(&info, caps))
            {
                gst_sample_unref(sample);
                return false;
            }

            GstMapInfo map{};
            if (!gst_buffer_map(buffer, &map, GST_MAP_READ))
            {
                gst_sample_unref(sample);
                return false;
            }

            frame.source        = uri_;
            frame.desc.width    = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
            frame.desc.height   = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
            frame.desc.channels = static_cast<int>(GST_VIDEO_INFO_N_COMPONENTS(&info));
            frame.desc.format   = pixel_format_from_caps(info);
            frame.pts           = GST_BUFFER_PTS_IS_VALID(buffer) ? static_cast<std::int64_t>(GST_BUFFER_PTS(buffer)) : 0;
            frame.data.assign(map.data, map.data + map.size);

            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            return true;
#else
            static_cast<void>(frame);
            return false;
#endif
        }

        void close()
        {
#if defined(CVKIT_WITH_GSTREAMER)
            if (pipeline_ != nullptr)
            {
                gst_element_set_state(pipeline_, GST_STATE_NULL);
            }
            if (sink_ != nullptr)
            {
                gst_object_unref(sink_);
                sink_ = nullptr;
            }
            if (pipeline_ != nullptr)
            {
                gst_object_unref(pipeline_);
                pipeline_ = nullptr;
            }
#endif
            open_   = false;
            primed_ = false;
            uri_.clear();
            backend_ = ReaderBackend::gstreamer;
        }

      private:
        std::string   uri_{};
        bool          open_{false};
        bool          primed_{false};
        ReaderBackend backend_{ReaderBackend::gstreamer};
#if defined(CVKIT_WITH_GSTREAMER)
        GstElement* pipeline_{nullptr};
        GstAppSink* sink_{nullptr};
#endif
    };

    Source::Source()
        : impl_(std::make_unique<Impl>())
    {
    }
    Source::~Source()                            = default;
    Source::Source(Source&&) noexcept            = default;
    Source& Source::operator=(Source&&) noexcept = default;

    bool    Source::open(std::string uri)
    {
        SourceOptions options{};
        options.uri     = std::move(uri);
        options.backend = ReaderBackend::gstreamer;
        return impl_->open(std::move(options));
    }

    bool Source::open(SourceOptions options)
    {
        return impl_->open(std::move(options));
    }

    bool Source::read(cvkit::core::Frame& frame)
    {
        return impl_->read(frame);
    }

    void Source::close()
    {
        impl_->close();
    }

}  // namespace cvkit::media
