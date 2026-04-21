#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cvkit/infer/model.h"

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
