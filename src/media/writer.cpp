#include "cvkit/media/writer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace cvkit::media
{
    namespace
    {

        [[nodiscard]] std::string make_status_message(WriterStatus status, std::string_view detail = {})
        {
            std::ostringstream stream;
            switch (status)
            {
                case WriterStatus::open:
                    stream << "open";
                    break;
                case WriterStatus::limit_reached:
                    stream << "limit reached";
                    break;
                case WriterStatus::invalid_options:
                    stream << "invalid options";
                    break;
                case WriterStatus::unsupported_backend:
                    stream << "unsupported backend";
                    break;
                case WriterStatus::backend_error:
                    stream << "backend error";
                    break;
                case WriterStatus::closed:
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

        [[nodiscard]] bool frame_matches_writer(const cvkit::core::Frame& frame, const WriterInfo& info)
        {
            return frame.desc.width == info.width &&
                   frame.desc.height == info.height &&
                   frame.desc.channels == 3 &&
                   frame.desc.format == cvkit::core::PixelFormat::bgr8 &&
                   frame.data.size() >= static_cast<std::size_t>(info.width * info.height * 3);
        }

        [[nodiscard]] int opencv_fourcc_for_uri(const std::string& uri)
        {
            auto extension = std::filesystem::path(uri).extension().string();
            std::transform(
                extension.begin(),
                extension.end(),
                extension.begin(),
                [](unsigned char ch)
                {
                    return static_cast<char>(std::tolower(ch));
                });

            if (extension == ".avi")
            {
                return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            }
            return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }

    }  // namespace

    class Writer::Impl
    {
      public:
        ~Impl()
        {
            close();
        }

        bool open(WriterOptions options)
        {
            close();
            if (options.uri.empty() || options.width <= 0 || options.height <= 0 || options.fps <= 0.0)
            {
                set_status(WriterStatus::invalid_options);
                return false;
            }

            if (options.backend == WriterBackend::ffmpeg || options.backend == WriterBackend::gstreamer)
            {
                set_status(WriterStatus::unsupported_backend, "only opencv writer is implemented");
                return false;
            }

            if (options.backend != WriterBackend::opencv)
            {
                set_status(WriterStatus::unsupported_backend);
                return false;
            }

            const auto fourcc = opencv_fourcc_for_uri(options.uri);
            writer_.open(options.uri, fourcc, options.fps, cv::Size(options.width, options.height), true);
            if (!writer_.isOpened())
            {
                set_status(WriterStatus::backend_error, "opencv failed to open writer");
                return false;
            }

            info_.uri        = std::move(options.uri);
            info_.backend    = options.backend;
            info_.open       = true;
            info_.width      = options.width;
            info_.height     = options.height;
            info_.fps        = options.fps;
            info_.max_frames = options.max_frames;
            set_status(WriterStatus::open);
            return true;
        }

        bool write(const cvkit::core::Frame& frame)
        {
            if (!info_.open)
            {
                return false;
            }
            if (info_.max_frames > 0 && info_.frame_count >= info_.max_frames)
            {
                set_status(WriterStatus::limit_reached);
                return false;
            }
            if (!frame_matches_writer(frame, info_))
            {
                set_status(WriterStatus::invalid_options, "frame does not match writer geometry or BGR layout");
                return false;
            }

            cv::Mat image(
                frame.desc.height,
                frame.desc.width,
                CV_8UC3,
                const_cast<std::uint8_t*>(frame.data.data()));
            writer_.write(image);
            ++info_.frame_count;
            set_status(WriterStatus::open);
            return true;
        }

        bool write(const cvkit::core::DeviceFrame& frame)
        {
            static_cast<void>(frame);
            if (!info_.open)
            {
                return false;
            }
            set_status(WriterStatus::unsupported_backend, "device frame writing is not implemented");
            return false;
        }

        void close()
        {
            writer_.release();
            info_ = {};
            set_status(WriterStatus::closed);
        }

        bool is_open() const
        {
            return info_.open;
        }

        WriterStatus status() const
        {
            return status_;
        }

        std::string_view status_message() const
        {
            return status_message_;
        }

        WriterInfo info() const
        {
            return info_;
        }

      private:
        void set_status(WriterStatus status, std::string_view detail = {})
        {
            status_         = status;
            status_message_ = make_status_message(status, detail);
        }

        cv::VideoWriter writer_{};
        WriterInfo      info_{};
        WriterStatus    status_{WriterStatus::closed};
        std::string     status_message_{make_status_message(WriterStatus::closed)};
    };

    Writer::Writer()
        : impl_(std::make_unique<Impl>())
    {
    }

    Writer::~Writer()                            = default;
    Writer::Writer(Writer&&) noexcept            = default;
    Writer& Writer::operator=(Writer&&) noexcept = default;

    bool Writer::open(WriterOptions options)
    {
        return impl_->open(std::move(options));
    }

    bool Writer::write(const cvkit::core::Frame& frame)
    {
        return impl_->write(frame);
    }

    bool Writer::write(const cvkit::core::DeviceFrame& frame)
    {
        return impl_->write(frame);
    }

    void Writer::close()
    {
        impl_->close();
    }

    bool Writer::is_open() const
    {
        return impl_->is_open();
    }

    WriterStatus Writer::status() const
    {
        return impl_->status();
    }

    std::string_view Writer::status_message() const
    {
        return impl_->status_message();
    }

    WriterInfo Writer::info() const
    {
        return impl_->info();
    }

}  // namespace cvkit::media
