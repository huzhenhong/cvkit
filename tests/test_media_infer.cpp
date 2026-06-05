#include <catch2/catch_test_macros.hpp>

#include "../src/infer/tasks/detection/yolo/yolo_preprocess.h"
#include "cvkit/infer/model.h"
#include "cvkit/infer/task_io.h"
#include "cvkit/media/source.h"

#include <cstdlib>
#include <filesystem>
#include <utility>
#include <vector>

namespace
{
    constexpr int kProcessCudaDeviceIndex = 0;
    constexpr int kPhysicalCudaDeviceIndexForProbe = 7;
}

TEST_CASE("gstreamer nvdec device frame feeds yolo cuda preprocess")
{
#if !defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    SKIP("CUDA preprocess kernel is required for media-to-infer device-frame integration");
#else
    const auto capabilities = cvkit::media::runtime_capabilities(kPhysicalCudaDeviceIndexForProbe);
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
    options.cuda_device_index = kProcessCudaDeviceIndex;

    cvkit::media::Source source;
    if (!source.open(std::move(options)))
    {
        INFO(source.status_message());
        SKIP("GStreamer CUDA source did not open in this environment");
    }

    cvkit::core::DeviceFrame frame{};
    REQUIRE(source.read(frame));
    REQUIRE(frame.valid());
    REQUIRE(frame.memory_device == cvkit::core::MemoryDevice::cuda);
    REQUIRE(frame.desc.format == cvkit::core::PixelFormat::nv12);
    CHECK(frame.device_index == kProcessCudaDeviceIndex);

    auto image = cvkit::infer::image_value_from_device_frame(frame);
    REQUIRE(image.has_valid_device_view());
    CHECK(image.storage_owner == frame.owner);

    cvkit::infer::TaskInput input{};
    input.add("image", image);

    const auto outcome = cvkit::infer::detail::preprocess_yolo(input, {1, 3, 640, 640}, true);
    REQUIRE(outcome.ok());
    CHECK(outcome.result.tensor.memory_device == cvkit::infer::MemoryDevice::cuda);
    CHECK(outcome.result.tensor.shape == std::vector<std::int64_t>{1, 3, 640, 640});
    CHECK(outcome.result.tensor.has_valid_device_view());
#endif
}

TEST_CASE("gstreamer nvdec device frame runs through tensorRT yolo detection")
{
#if !defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    SKIP("CUDA preprocess kernel is required for media-to-infer device-frame integration");
#else
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to run TensorRT media-to-infer smoke validation");
    }

    const auto capabilities = cvkit::media::runtime_capabilities(kPhysicalCudaDeviceIndexForProbe);
    if (!capabilities.gstreamer || !capabilities.gstreamer_nvh264dec)
    {
        SKIP("GStreamer NVDEC support is not available in this build");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto video_path  = source_root / "assets" / "video" / "test.mp4";
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    if (!std::filesystem::exists(video_path) ||
        !std::filesystem::exists(model_path) ||
        !std::filesystem::exists(labels_path))
    {
        SKIP("required media/model assets are not present");
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path.string();
    spec.labels_path  = labels_path.string();
    spec.backend      = cvkit::infer::Backend::tensorrt;
    spec.task         = cvkit::infer::TaskKind::detection;
    spec.family       = "yolo11";
    spec.cache_policy = cvkit::infer::CachePolicy::default_policy;
    spec.device       = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, kProcessCudaDeviceIndex};
    spec.tensorrt_profiles.push_back({
        .input_name = "images",
        .shape =
            {
                .min = {1, 3, 320, 320},
                .opt = {1, 3, 640, 640},
                .max = {1, 3, 1280, 1280},
            },
    });

    cvkit::infer::Model model;
    REQUIRE(model.load(spec));
    REQUIRE(model.loaded());

    cvkit::media::SourceOptions options{};
    options.uri               = video_path.string();
    options.backend           = cvkit::media::ReaderBackend::gstreamer;
    options.output_memory     = cvkit::media::SourceMemory::cuda;
    options.cuda_device_index = kProcessCudaDeviceIndex;

    cvkit::media::Source source;
    if (!source.open(std::move(options)))
    {
        INFO(source.status_message());
        SKIP("GStreamer CUDA source did not open in this environment");
    }

    cvkit::core::DeviceFrame frame{};
    REQUIRE(source.read(frame));
    REQUIRE(frame.valid());
    REQUIRE(frame.memory_device == cvkit::core::MemoryDevice::cuda);
    REQUIRE(frame.desc.format == cvkit::core::PixelFormat::nv12);
    CHECK(frame.device_index == kProcessCudaDeviceIndex);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::image_value_from_device_frame(frame));

    const auto  output     = model.run_sync(input);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);

    const auto trace = model.last_graph_trace();
    REQUIRE_FALSE(trace.empty());
    CHECK(trace.front().ok);
    CHECK(trace.front().message.empty());
#endif
}
