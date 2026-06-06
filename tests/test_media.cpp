#include <catch2/catch_test_macros.hpp>

#include "cvkit/media/source.h"
#include "cvkit/media/writer.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>

TEST_CASE("media source rejects empty uri with inspectable status")
{
    cvkit::media::Source source;
    CHECK_FALSE(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::closed);

    CHECK_FALSE(source.open(std::string{}));
    CHECK_FALSE(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::invalid_uri);
    CHECK_FALSE(source.status_message().empty());
}

TEST_CASE("media source reports unsupported backend")
{
    cvkit::media::SourceOptions options{};
    options.uri = "assets/images/face.jpg";
    options.backend = cvkit::media::ReaderBackend::ffmpeg;

    cvkit::media::Source source;
    CHECK_FALSE(source.open(std::move(options)));
    CHECK_FALSE(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::unsupported_backend);
}

TEST_CASE("opencv media source reads image file as host bgr frame")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto image_path = source_root / "assets" / "images" / "face.jpg";
    if (!std::filesystem::exists(image_path))
    {
        SKIP("face.jpg is not present under assets");
    }

    cvkit::media::SourceOptions options{};
    options.uri = image_path.string();
    options.backend = cvkit::media::ReaderBackend::opencv;

    cvkit::media::Source source;
    REQUIRE(source.open(std::move(options)));
    CHECK(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::open);

    const auto opened_info = source.info();
    CHECK(opened_info.open);
    CHECK(opened_info.backend == cvkit::media::ReaderBackend::opencv);
    CHECK(opened_info.width == 2048);
    CHECK(opened_info.height == 1150);

    cvkit::core::Frame frame{};
    REQUIRE(source.read(frame));
    CHECK(source.status() == cvkit::media::SourceStatus::open);
    CHECK(frame.desc.width == 2048);
    CHECK(frame.desc.height == 1150);
    CHECK(frame.desc.channels == 3);
    CHECK(frame.desc.format == cvkit::core::PixelFormat::bgr8);
    CHECK(frame.source == image_path.string());
    CHECK(frame.data.size() == static_cast<std::size_t>(2048 * 1150 * 3));

    const auto read_info = source.info();
    CHECK(read_info.open);
    CHECK(read_info.width == frame.desc.width);
    CHECK(read_info.height == frame.desc.height);
    CHECK(read_info.channels == frame.desc.channels);
    CHECK(read_info.format == frame.desc.format);

    CHECK_FALSE(source.read(frame));
    CHECK_FALSE(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::end_of_stream);

    source.close();
    CHECK_FALSE(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::closed);
}

TEST_CASE("media source string overload defaults to opencv backend")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto image_path = source_root / "assets" / "images" / "face.jpg";
    if (!std::filesystem::exists(image_path))
    {
        SKIP("face.jpg is not present under assets");
    }

    cvkit::media::Source source;
    REQUIRE(source.open(image_path.string()));
    CHECK(source.info().backend == cvkit::media::ReaderBackend::opencv);

    cvkit::core::Frame frame{};
    REQUIRE(source.read(frame));
    CHECK(frame.desc.width == 2048);
    CHECK(frame.desc.height == 1150);
}

TEST_CASE("opencv media source reads video file with metadata")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto video_path = source_root / "assets" / "video" / "test.mp4";
    if (!std::filesystem::exists(video_path))
    {
        SKIP("test.mp4 is not present under assets/video");
    }

    cvkit::media::SourceOptions options{};
    options.uri = video_path.string();
    options.backend = cvkit::media::ReaderBackend::opencv;

    cvkit::media::Source source;
    REQUIRE(source.open(std::move(options)));
    CHECK(source.is_open());
    CHECK(source.status() == cvkit::media::SourceStatus::open);

    const auto opened_info = source.info();
    CHECK(opened_info.open);
    CHECK(opened_info.width > 0);
    CHECK(opened_info.height > 0);
    CHECK(opened_info.channels == 3);
    CHECK(opened_info.format == cvkit::core::PixelFormat::bgr8);
    CHECK(opened_info.fps > 0.0);
    CHECK(opened_info.frame_count > 0);
    CHECK(opened_info.frame_index == 0);

    std::int64_t previous_pts = -1;
    for (int i = 0; i < 8; ++i)
    {
        cvkit::core::Frame frame{};
        REQUIRE(source.read(frame));
        CHECK(frame.desc.width == opened_info.width);
        CHECK(frame.desc.height == opened_info.height);
        CHECK(frame.desc.channels == 3);
        CHECK(frame.desc.format == cvkit::core::PixelFormat::bgr8);
        CHECK_FALSE(frame.data.empty());
        CHECK(frame.source == video_path.string());
        if (previous_pts >= 0 && frame.pts > 0)
        {
            CHECK(frame.pts >= previous_pts);
        }
        previous_pts = frame.pts;
    }

    const auto read_info = source.info();
    CHECK(read_info.open);
    CHECK(read_info.frame_index >= 8);
}

TEST_CASE("media runtime reports gstreamer cuda decode capabilities")
{
    const auto capabilities = cvkit::media::runtime_capabilities(7);

    if (!capabilities.gstreamer)
    {
        SKIP("GStreamer support is not enabled in this build");
    }

    CHECK(capabilities.gstreamer_appsink);
    CHECK(capabilities.gstreamer_decodebin);
    CHECK(capabilities.gstreamer_h264parse);

    INFO("nvh264dec=" << capabilities.gstreamer_nvh264dec
                      << " nvh264device7dec=" << capabilities.gstreamer_nvh264_device_decoder
                      << " cudaupload=" << capabilities.gstreamer_cudaupload
                      << " cudadownload=" << capabilities.gstreamer_cudadownload
                      << " cudaconvert=" << capabilities.gstreamer_cuda_convert);
}

TEST_CASE("gstreamer media source reads h264 video as cuda device frame")
{
    const auto capabilities = cvkit::media::runtime_capabilities(7);
    if (!capabilities.gstreamer || !capabilities.gstreamer_nvh264dec)
    {
        SKIP("GStreamer NVDEC support is not available in this build");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto video_path = source_root / "assets" / "video" / "test.mp4";
    if (!std::filesystem::exists(video_path))
    {
        SKIP("test.mp4 is not present under assets/video");
    }

    cvkit::media::SourceOptions options{};
    options.uri = video_path.string();
    options.backend = cvkit::media::ReaderBackend::gstreamer;
    options.output_memory = cvkit::media::SourceMemory::cuda;
    options.cuda_device_index = 7;

    cvkit::media::Source source;
    if (!source.open(std::move(options)))
    {
        INFO(source.status_message());
        SKIP("GStreamer CUDA source did not open in this environment");
    }

    cvkit::core::Frame host_frame{};
    CHECK_FALSE(source.read(host_frame));
    CHECK(source.status() == cvkit::media::SourceStatus::unsupported_backend);

    cvkit::core::DeviceFrame frame{};
    REQUIRE(source.read(frame));
    CHECK(frame.valid());
    CHECK(frame.memory_device == cvkit::core::MemoryDevice::cuda);
    CHECK(frame.desc.width == 2560);
    CHECK(frame.desc.height == 1440);
    CHECK(frame.desc.channels == 1);
    CHECK(frame.desc.format == cvkit::core::PixelFormat::nv12);
    CHECK(frame.bytes > 0);
    CHECK(frame.stride_bytes > 0);
    CHECK(frame.device_index == 7);
    CHECK(frame.source == video_path.string());
}

TEST_CASE("opencv media writer writes host bgr frames")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto output_dir  = source_root / "assets" / "output";
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / "test_media_writer.avi";
    std::filesystem::remove(output_path);

    cvkit::media::WriterOptions options{};
    options.uri        = output_path.string();
    options.backend    = cvkit::media::WriterBackend::opencv;
    options.width      = 64;
    options.height     = 48;
    options.fps        = 12.0;
    options.max_frames = 3;

    cvkit::media::Writer writer;
    REQUIRE(writer.open(std::move(options)));
    CHECK(writer.is_open());
    CHECK(writer.status() == cvkit::media::WriterStatus::open);

    cvkit::core::Frame frame{};
    frame.desc.width    = 64;
    frame.desc.height   = 48;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(64 * 48 * 3), 64U);

    REQUIRE(writer.write(frame));
    frame.data.assign(static_cast<std::size_t>(64 * 48 * 3), 128U);
    REQUIRE(writer.write(frame));
    frame.data.assign(static_cast<std::size_t>(64 * 48 * 3), 192U);
    REQUIRE(writer.write(frame));
    CHECK(writer.info().frame_count == 3);
    CHECK_FALSE(writer.write(frame));
    CHECK(writer.status() == cvkit::media::WriterStatus::limit_reached);
    writer.close();
    CHECK_FALSE(writer.is_open());

    REQUIRE(std::filesystem::exists(output_path));

    cvkit::media::Source source;
    REQUIRE(source.open(output_path.string()));
    cvkit::core::Frame decoded{};
    REQUIRE(source.read(decoded));
    CHECK(decoded.desc.width == 64);
    CHECK(decoded.desc.height == 48);
    CHECK(decoded.desc.channels == 3);
    CHECK(decoded.desc.format == cvkit::core::PixelFormat::bgr8);
}

TEST_CASE("media writer rejects unsupported and device-frame paths explicitly")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto output_dir  = source_root / "assets" / "output";
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / "device_writer_reject.avi";
    std::filesystem::remove(output_path);

    cvkit::media::WriterOptions options{};
    options.uri     = output_path.string();
    options.backend = cvkit::media::WriterBackend::gstreamer;
    options.width   = 64;
    options.height  = 48;
    options.fps     = 25.0;

    cvkit::media::Writer writer;
    CHECK_FALSE(writer.open(std::move(options)));
    CHECK(writer.status() == cvkit::media::WriterStatus::unsupported_backend);

    cvkit::media::WriterOptions host_options{};
    host_options.uri     = output_path.string();
    host_options.backend = cvkit::media::WriterBackend::opencv;
    host_options.width   = 64;
    host_options.height  = 48;
    host_options.fps     = 25.0;
    REQUIRE(writer.open(std::move(host_options)));

    cvkit::core::DeviceFrame frame{};
    frame.desc.width    = 64;
    frame.desc.height   = 48;
    frame.desc.channels = 1;
    frame.desc.format   = cvkit::core::PixelFormat::nv12;
    frame.memory_device = cvkit::core::MemoryDevice::cuda;
    CHECK_FALSE(writer.write(frame));
    CHECK(writer.status() == cvkit::media::WriterStatus::unsupported_backend);
}
