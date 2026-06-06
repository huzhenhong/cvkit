#include "cvkit/media/source.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#if defined(CVKIT_WITH_GSTREAMER)
    #include <gst/app/gstappsink.h>
    #include <gst/video/video.h>
    #include <gst/gst.h>
#endif

#include <filesystem>
#include <iostream>
#include <sstream>
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

        [[nodiscard]] std::string make_cuda_h264_pipeline_description(const std::string& uri)
        {
            if (uri.find("://") == std::string::npos && std::filesystem::exists(uri))
            {
                return "filesrc location=\"" + escape_location(uri) + "\" ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! nvh264dec ! video/x-raw(memory:CUDAMemory),format=NV12 ! appsink name=cvkit_sink sync=false";
            }

            return "uridecodebin uri=\"" + normalize_uri(uri) + "\" ! h264parse ! nvh264dec ! video/x-raw(memory:CUDAMemory),format=NV12 ! appsink name=cvkit_sink sync=false";
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
                case GST_VIDEO_FORMAT_NV12:
                    return cvkit::core::PixelFormat::nv12;
                default:
                    return cvkit::core::PixelFormat::unknown;
            }
        }

        [[nodiscard]] cvkit::core::MemoryDevice memory_device_from_caps(const GstCaps* caps)
        {
            if (caps == nullptr || gst_caps_is_empty(caps))
            {
                return cvkit::core::MemoryDevice::host;
            }

            const auto* features = gst_caps_get_features(caps, 0);
            if (features != nullptr && gst_caps_features_contains(features, "memory:CUDAMemory"))
            {
                return cvkit::core::MemoryDevice::cuda;
            }
            return cvkit::core::MemoryDevice::host;
        }

        struct MappedGstSample
        {
            GstSample* sample{nullptr};
            GstBuffer* buffer{nullptr};
            GstMapInfo map{};

            ~MappedGstSample()
            {
                if (buffer != nullptr)
                {
                    gst_buffer_unmap(buffer, &map);
                }
                if (sample != nullptr)
                {
                    gst_sample_unref(sample);
                }
            }
        };

        [[nodiscard]] bool has_element_factory(const char* name)
        {
            auto* factory = gst_element_factory_find(name);
            if (factory == nullptr)
            {
                return false;
            }
            gst_object_unref(factory);
            return true;
        }

    }  // namespace
#endif

    namespace
    {

        [[nodiscard]] std::string make_status_message(SourceStatus status, std::string_view detail = {})
        {
            std::ostringstream stream;
            switch (status)
            {
                case SourceStatus::open:
                    stream << "open";
                    break;
                case SourceStatus::end_of_stream:
                    stream << "end of stream";
                    break;
                case SourceStatus::invalid_uri:
                    stream << "invalid uri";
                    break;
                case SourceStatus::unsupported_backend:
                    stream << "unsupported backend";
                    break;
                case SourceStatus::backend_error:
                    stream << "backend error";
                    break;
                case SourceStatus::closed:
                default:
                    stream << "closed";
                    break;
            }
            if (!detail.empty())
            {
                stream << ": " << detail;
            }
            return stream.str();
        }

        [[nodiscard]] cvkit::core::Frame frame_from_mat(const cv::Mat& image, const std::string& source)
        {
            cv::Mat bgr;
            if (image.channels() == 3)
            {
                bgr = image;
            }
            else if (image.channels() == 4)
            {
                cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
            }
            else if (image.channels() == 1)
            {
                cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
            }
            else
            {
                return {};
            }

            if (!bgr.isContinuous())
            {
                bgr = bgr.clone();
            }

            cvkit::core::Frame frame{};
            frame.source        = source;
            frame.desc.width    = bgr.cols;
            frame.desc.height   = bgr.rows;
            frame.desc.channels = bgr.channels();
            frame.desc.format   = cvkit::core::PixelFormat::bgr8;
            frame.data.assign(bgr.data, bgr.data + bgr.total() * bgr.elemSize());
            return frame;
        }

        void update_info_from_frame(SourceInfo& info, const cvkit::core::Frame& frame)
        {
            info.width    = frame.desc.width;
            info.height   = frame.desc.height;
            info.channels = frame.desc.channels;
            info.format   = frame.desc.format;
        }

    }  // namespace

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
            backend_           = options.backend;
            output_memory_     = options.output_memory;
            cuda_device_index_ = options.cuda_device_index;
            uri_               = std::move(options.uri);
            info_.uri               = uri_;
            info_.backend           = backend_;
            info_.output_memory     = output_memory_;
            info_.cuda_device_index = cuda_device_index_;
            if (uri_.empty())
            {
                set_status(SourceStatus::invalid_uri);
                return false;
            }

            if (backend_ == ReaderBackend::opencv)
            {
                if (output_memory_ != SourceMemory::host)
                {
                    set_status(SourceStatus::unsupported_backend, "opencv source only supports host output");
                    return false;
                }
                return open_opencv();
            }

#if defined(CVKIT_WITH_GSTREAMER)
            if (backend_ == ReaderBackend::gstreamer)
            {
                ensure_gstreamer_initialized();
                const auto pipeline_description = output_memory_ == SourceMemory::cuda
                                                      ? make_cuda_h264_pipeline_description(uri_)
                                                      : make_pipeline_description(uri_);

                GError* error = nullptr;
                pipeline_     = gst_parse_launch(pipeline_description.c_str(), &error);
                if (pipeline_ == nullptr)
                {
                    if (error != nullptr)
                    {
                        set_status(SourceStatus::backend_error, error->message);
                        g_error_free(error);
                    }
                    else
                    {
                        set_status(SourceStatus::backend_error, "failed to create gstreamer pipeline");
                    }
                    return false;
                }

                sink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "cvkit_sink"));
                if (sink_ == nullptr)
                {
                    close();
                    set_status(SourceStatus::backend_error, "gstreamer appsink not found");
                    return false;
                }

                gst_app_sink_set_emit_signals(sink_, FALSE);
                gst_app_sink_set_drop(sink_, FALSE);
                gst_app_sink_set_max_buffers(sink_, 4);

                const auto state_change = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
                if (state_change == GST_STATE_CHANGE_FAILURE)
                {
                    close();
                    set_status(SourceStatus::backend_error, "failed to start gstreamer pipeline");
                    return false;
                }

                GstState current = GST_STATE_NULL;
                GstState pending = GST_STATE_NULL;
                gst_element_get_state(pipeline_, &current, &pending, 5 * GST_SECOND);

                if (!check_bus_for_error(pipeline_))
                {
                    close();
                    set_status(SourceStatus::backend_error, "gstreamer bus reported an error");
                    return false;
                }

                open_ = true;
                info_.open = true;
                set_status(SourceStatus::open);
                return true;
            }
#endif

            set_status(SourceStatus::unsupported_backend);
            return false;
        }

        bool read(cvkit::core::Frame& frame)
        {
            if (!open_)
            {
                return false;
            }

            if (backend_ == ReaderBackend::opencv)
            {
                return read_opencv(frame);
            }

#if defined(CVKIT_WITH_GSTREAMER)
            if (output_memory_ != SourceMemory::host)
            {
                set_status(SourceStatus::unsupported_backend, "use read(DeviceFrame&) for cuda source output");
                return false;
            }

            GstSample* sample = nullptr;
            if (!primed_)
            {
                sample  = gst_app_sink_try_pull_preroll(sink_, 15 * GST_SECOND);
                primed_ = true;
                if (sample == nullptr)
                {
                    sample = gst_app_sink_try_pull_sample(sink_, 15 * GST_SECOND);
                }
            }
            else
            {
                sample = gst_app_sink_try_pull_sample(sink_, 15 * GST_SECOND);
            }

            if (sample == nullptr)
            {
                if (check_bus_for_error(pipeline_))
                {
                    set_status(SourceStatus::end_of_stream);
                }
                else
                {
                    set_status(SourceStatus::backend_error, "gstreamer bus reported an error");
                }
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
            update_info_from_frame(info_, frame);
            ++info_.frame_index;

            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            return true;
#endif
            static_cast<void>(frame);
            set_status(SourceStatus::unsupported_backend);
            return false;
        }

        bool read(cvkit::core::DeviceFrame& frame)
        {
            if (!open_)
            {
                return false;
            }

            if (backend_ != ReaderBackend::gstreamer || output_memory_ != SourceMemory::cuda)
            {
                set_status(SourceStatus::unsupported_backend, "device frame output requires gstreamer cuda source output");
                return false;
            }

#if defined(CVKIT_WITH_GSTREAMER)
            GstSample* sample = nullptr;
            if (!primed_)
            {
                sample  = gst_app_sink_try_pull_preroll(sink_, 15 * GST_SECOND);
                primed_ = true;
                if (sample == nullptr)
                {
                    sample = gst_app_sink_try_pull_sample(sink_, 15 * GST_SECOND);
                }
            }
            else
            {
                sample = gst_app_sink_try_pull_sample(sink_, 15 * GST_SECOND);
            }

            if (sample == nullptr)
            {
                if (check_bus_for_error(pipeline_))
                {
                    set_status(SourceStatus::end_of_stream);
                }
                else
                {
                    set_status(SourceStatus::backend_error, "gstreamer bus reported an error");
                }
                return false;
            }

            auto* caps   = gst_sample_get_caps(sample);
            auto* buffer = gst_sample_get_buffer(sample);
            if (caps == nullptr || buffer == nullptr)
            {
                gst_sample_unref(sample);
                set_status(SourceStatus::backend_error, "gstreamer sample is missing caps or buffer");
                return false;
            }

            GstVideoInfo info{};
            if (!gst_video_info_from_caps(&info, caps))
            {
                gst_sample_unref(sample);
                set_status(SourceStatus::backend_error, "failed to parse gstreamer video caps");
                return false;
            }

            auto owner = std::make_shared<MappedGstSample>();
            owner->sample = sample;
            owner->buffer = buffer;
            if (!gst_buffer_map(buffer, &owner->map, GST_MAP_READ))
            {
                owner->buffer = nullptr;
                set_status(SourceStatus::backend_error, "failed to map gstreamer cuda buffer");
                return false;
            }

            frame.source        = uri_;
            frame.desc.width    = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
            frame.desc.height   = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
            frame.desc.channels = 1;
            frame.desc.format   = pixel_format_from_caps(info);
            frame.data          = reinterpret_cast<std::uintptr_t>(owner->map.data);
            frame.bytes         = owner->map.size;
            frame.stride_bytes  = static_cast<std::size_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
            frame.memory_device = memory_device_from_caps(caps);
            frame.device_index  = cuda_device_index_;
            frame.pts           = GST_BUFFER_PTS_IS_VALID(buffer) ? static_cast<std::int64_t>(GST_BUFFER_PTS(buffer)) : 0;
            frame.owner         = std::move(owner);

            info_.width    = frame.desc.width;
            info_.height   = frame.desc.height;
            info_.channels = frame.desc.channels;
            info_.format   = frame.desc.format;
            ++info_.frame_index;
            set_status(SourceStatus::open);
            return true;
#else
            static_cast<void>(frame);
            set_status(SourceStatus::unsupported_backend);
            return false;
#endif
        }

        void close()
        {
            capture_.release();
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
            image_mode_ = false;
            image_consumed_ = false;
            image_.release();
            uri_.clear();
            backend_ = ReaderBackend::opencv;
            output_memory_ = SourceMemory::host;
            cuda_device_index_ = 0;
            info_ = {};
            set_status(SourceStatus::closed);
        }

        bool is_open() const
        {
            return open_;
        }

        SourceStatus status() const
        {
            return status_;
        }

        std::string_view status_message() const
        {
            return status_message_;
        }

        SourceInfo info() const
        {
            return info_;
        }

      private:
        bool open_opencv()
        {
            image_ = cv::imread(uri_, cv::IMREAD_UNCHANGED);
            if (!image_.empty())
            {
                image_mode_ = true;
                image_consumed_ = false;
                open_ = true;
                info_.open = true;
                info_.width = image_.cols;
                info_.height = image_.rows;
                info_.channels = 3;
                info_.format = cvkit::core::PixelFormat::bgr8;
                info_.fps = 0.0;
                info_.frame_count = 1;
                info_.frame_index = 0;
                set_status(SourceStatus::open);
                return true;
            }

            capture_.open(uri_);
            if (!capture_.isOpened())
            {
                set_status(SourceStatus::backend_error, "opencv failed to open source");
                return false;
            }

            open_ = true;
            info_.open = true;
            info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            info_.channels = 3;
            info_.format = cvkit::core::PixelFormat::bgr8;
            info_.fps = capture_.get(cv::CAP_PROP_FPS);
            info_.frame_count = static_cast<std::int64_t>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
            info_.frame_index = 0;
            set_status(SourceStatus::open);
            return true;
        }

        bool read_opencv(cvkit::core::Frame& frame)
        {
            if (image_mode_)
            {
                if (image_consumed_)
                {
                    open_ = false;
                    info_.open = false;
                    set_status(SourceStatus::end_of_stream);
                    return false;
                }

                frame = frame_from_mat(image_, uri_);
                if (frame.data.empty())
                {
                    set_status(SourceStatus::backend_error, "unsupported opencv image format");
                    return false;
                }

                image_consumed_ = true;
                update_info_from_frame(info_, frame);
                info_.frame_index = 1;
                set_status(SourceStatus::open);
                return true;
            }

            cv::Mat image;
            if (!capture_.read(image) || image.empty())
            {
                open_ = false;
                info_.open = false;
                set_status(SourceStatus::end_of_stream);
                return false;
            }

            frame = frame_from_mat(image, uri_);
            if (frame.data.empty())
            {
                set_status(SourceStatus::backend_error, "unsupported opencv frame format");
                return false;
            }

            const auto pos_msec = capture_.get(cv::CAP_PROP_POS_MSEC);
            frame.pts = pos_msec > 0.0 ? static_cast<std::int64_t>(pos_msec * 1000000.0) : 0;
            update_info_from_frame(info_, frame);
            info_.frame_index = static_cast<std::int64_t>(capture_.get(cv::CAP_PROP_POS_FRAMES));
            set_status(SourceStatus::open);
            return true;
        }

        void set_status(SourceStatus status, std::string_view detail = {})
        {
            status_ = status;
            status_message_ = make_status_message(status, detail);
        }

        std::string   uri_{};
        bool          open_{false};
        bool          primed_{false};
        ReaderBackend backend_{ReaderBackend::opencv};
        SourceMemory  output_memory_{SourceMemory::host};
        int           cuda_device_index_{0};
        SourceStatus  status_{SourceStatus::closed};
        std::string   status_message_{make_status_message(SourceStatus::closed)};
        SourceInfo    info_{};
        cv::VideoCapture capture_{};
        cv::Mat       image_{};
        bool          image_mode_{false};
        bool          image_consumed_{false};
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
        options.backend = ReaderBackend::opencv;
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

    bool Source::read(cvkit::core::DeviceFrame& frame)
    {
        return impl_->read(frame);
    }

    void Source::close()
    {
        impl_->close();
    }

    bool Source::is_open() const
    {
        return impl_->is_open();
    }

    SourceStatus Source::status() const
    {
        return impl_->status();
    }

    std::string_view Source::status_message() const
    {
        return impl_->status_message();
    }

    SourceInfo Source::info() const
    {
        return impl_->info();
    }

    RuntimeCapabilities runtime_capabilities(int cuda_device_index)
    {
        RuntimeCapabilities capabilities{};
#if defined(CVKIT_WITH_GSTREAMER)
        ensure_gstreamer_initialized();

        capabilities.gstreamer = true;
        capabilities.gstreamer_appsink = has_element_factory("appsink");
        capabilities.gstreamer_decodebin = has_element_factory("decodebin");
        capabilities.gstreamer_h264parse = has_element_factory("h264parse");
        capabilities.gstreamer_avdec_h264 = has_element_factory("avdec_h264");
        capabilities.gstreamer_nvh264dec = has_element_factory("nvh264dec");
        capabilities.gstreamer_cudaupload = has_element_factory("cudaupload");
        capabilities.gstreamer_cudadownload = has_element_factory("cudadownload");
        capabilities.gstreamer_cuda_convert = has_element_factory("cudaconvert");
        capabilities.gstreamer_appsrc = has_element_factory("appsrc");
        capabilities.gstreamer_videoconvert = has_element_factory("videoconvert");
        capabilities.gstreamer_jpegenc = has_element_factory("jpegenc");
        capabilities.gstreamer_avimux = has_element_factory("avimux");
        capabilities.gstreamer_x264enc = has_element_factory("x264enc");
        capabilities.gstreamer_mp4mux = has_element_factory("mp4mux");
        capabilities.gstreamer_nvh264enc = has_element_factory("nvh264enc");

        if (cuda_device_index >= 0)
        {
            const auto device_decoder_name = std::string("nvh264device") + std::to_string(cuda_device_index) + "dec";
            capabilities.gstreamer_nvh264_device_decoder = has_element_factory(device_decoder_name.c_str());
        }
#else
        static_cast<void>(cuda_device_index);
#endif
        return capabilities;
    }

}  // namespace cvkit::media
