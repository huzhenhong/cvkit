#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cvkit/infer/model.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>

TEST_CASE("model load fails for missing onnx file")
{
    cvkit::infer::Model model;
    CHECK_FALSE(model.load("missing.onnx"));
    CHECK_FALSE(model.loaded());
}

TEST_CASE("model loads coco labels from standard asset file")
{
    const auto          source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto          labels_path = source_root / "assets" / "labels" / "coco80.txt";

    cvkit::infer::Model model;
    REQUIRE(model.load_labels(labels_path.string()));
    CHECK(model.labels_path() == labels_path.string());
}

TEST_CASE("model exposes configurable yolo thresholds")
{
    cvkit::infer::Model model;
    CHECK(model.confidence_threshold() == Catch::Approx(0.25F));
    CHECK(model.iou_threshold() == Catch::Approx(0.45F));

    model.set_confidence_threshold(0.6F);
    model.set_iou_threshold(0.3F);

    CHECK(model.confidence_threshold() == Catch::Approx(0.6F));
    CHECK(model.iou_threshold() == Catch::Approx(0.3F));
}

TEST_CASE("model submit runs inference asynchronously and returns detections")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(labels_path));
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::Backend backend = cvkit::infer::Backend::none;
#if defined(CVKIT_WITH_ONNXRUNTIME)
    backend = cvkit::infer::Backend::onnxruntime;
#elif defined(CVKIT_WITH_TENSORRT)
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to enable async TensorRT smoke validation");
    }
    backend = cvkit::infer::Backend::tensorrt;
#else
    SKIP("no inference backend is enabled in this build");
#endif

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path.string();
    spec.labels_path  = labels_path.string();
    spec.backend      = backend;
    spec.task         = cvkit::infer::TaskKind::detection;
    spec.family       = "yolo11";
    spec.cache_policy = cvkit::infer::CachePolicy::default_policy;

    cvkit::infer::Model model;
    REQUIRE(model.load(spec));
    REQUIRE(model.loaded());
    REQUIRE(model.backend() == backend);

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();
    frame.data.assign(image.data, image.data + image.total() * image.elemSize());

    cvkit::infer::TaskInput input{};
    input.add("image", frame);

    const auto sync_output = model.run_sync(input);
    const auto* sync_detections = sync_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(sync_detections != nullptr);
    REQUIRE_FALSE(sync_detections->empty());

    auto future = model.submit(input);
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);

    const auto async_output = future.get();
    const auto* async_detections = async_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(async_detections != nullptr);
    CHECK(async_detections->size() == sync_detections->size());
}

#if defined(CVKIT_WITH_TENSORRT)
TEST_CASE("tensorrt backend loads yolo11 model and returns detections for sample image")
{
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to enable the long-running TensorRT smoke test");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(labels_path));
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path  = model_path.string();
    spec.labels_path = labels_path.string();
    spec.backend     = cvkit::infer::Backend::tensorrt;
    spec.task        = cvkit::infer::TaskKind::detection;
    spec.family      = "yolo11";

    cvkit::infer::Model model;
    REQUIRE(model.load(spec));
    REQUIRE(model.loaded());
    REQUIRE(model.backend() == cvkit::infer::Backend::tensorrt);

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();
    frame.data.assign(image.data, image.data + image.total() * image.elemSize());

    const auto detections = model.run_detection(frame);
    CHECK_FALSE(detections.empty());
}
#endif
