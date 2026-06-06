#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/infer/backends/backend_session.h"
#include "../src/infer/graph/graph.h"
#include "../src/infer/tasks/classification/classification_pipeline.h"
#include "../src/infer/tasks/detection/yolo/yolo_preprocess.h"
#include "../src/infer/tasks/face_detection/face_detection_pipeline.h"
#include "../src/infer/tasks/facemesh/facemesh_pipeline.h"
#include "../src/infer/tasks/pose/pose_pipeline.h"
#include "../src/infer/tasks/promptable_segmentation/promptable_preprocess_cuda.h"
#include "../src/infer/tasks/segmentation/segmentation_pipeline.h"
#include "cvkit/infer/debug.h"
#include "cvkit/infer/model.h"
#include "cvkit/infer/tensor_io.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iterator>
#include <memory>

#if defined(CVKIT_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

namespace
{

    class TestNode final : public cvkit::infer::detail::INode
    {
      public:
        explicit TestNode(
            std::string_view         node_name,
            std::vector<std::string> consumes = {},
            std::vector<std::string> produces = {})
            : name_(node_name)
            , consumes_(std::move(consumes))
            , produces_(std::move(produces))
        {
        }

        [[nodiscard]] std::string_view name() const override
        {
            return name_;
        }

        [[nodiscard]] std::vector<std::string> consumes() const override
        {
            return consumes_;
        }

        [[nodiscard]] std::vector<std::string> produces() const override
        {
            return produces_;
        }

        [[nodiscard]] cvkit::infer::detail::Packet process(cvkit::infer::detail::Packet packet) const override
        {
            packet.output.add("node", std::string{name_});
            packet.put(std::string{name_}, static_cast<int>(packet.output.items.size()));
            return packet;
        }

      private:
        std::string              name_{};
        std::vector<std::string> consumes_{};
        std::vector<std::string> produces_{};
    };

    class StubBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::none;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static_cast<void>(index);
            return nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static_cast<void>(index);
            return nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            static_cast<void>(inputs);
            return {};
        }
    };

    class StubClassificationBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "images",
                {1, 3, 224, 224},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo output{
                "logits",
                {1, 3},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &output : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 224, 224});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensor output{};
            output.name  = "logits";
            output.shape = {1, 3};
            output.data  = {0.1F, 0.85F, 0.05F};
            return {std::move(output)};
        }
    };

    class StubSegmentationBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "images",
                {1, 3, 4, 4},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo output{
                "logits",
                {1, 2, 4, 4},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &output : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 4, 4});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensor output{};
            output.name  = "logits";
            output.shape = {1, 2, 4, 4};
            output.data.resize(2U * 4U * 4U, 0.0F);

            const auto plane_size = 4U * 4U;
            for (std::size_t index = 0; index < plane_size; ++index)
            {
                output.data[index]              = 0.1F;
                output.data[plane_size + index] = index % 2U == 0U ? 0.9F : 0.0F;
            }
            return {std::move(output)};
        }
    };

    class StubFaceDetectionBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "images",
                {1, 3, 64, 64},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo output{
                "faces",
                {1, 2, 15},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &output : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 64, 64});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensor output{};
            output.name  = "faces";
            output.shape = {1, 2, 15};
            output.data  = {
                10.0F, 12.0F, 30.0F, 36.0F, 0.95F,
                14.0F, 18.0F, 26.0F, 18.0F, 20.0F, 24.0F, 15.0F, 30.0F, 25.0F, 30.0F,
                11.0F, 13.0F, 29.0F, 35.0F, 0.50F,
                14.0F, 18.0F, 26.0F, 18.0F, 20.0F, 24.0F, 15.0F, 30.0F, 25.0F, 30.0F};
            return {std::move(output)};
        }
    };

    class StubScrfdBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "input.1",
                {1, 3, 64, 64},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo outputs[] = {
                {"score_8", {1, 128, 1}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"score_16", {1, 32, 1}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"score_32", {1, 8, 1}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"bbox_8", {1, 128, 4}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"bbox_16", {1, 32, 4}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"bbox_32", {1, 8, 4}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"kps_8", {1, 128, 10}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"kps_16", {1, 32, 10}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
                {"kps_32", {1, 8, 10}, cvkit::infer::TensorDataType::float32, cvkit::infer::MemoryDevice::host},
            };
            return index < std::size(outputs) ? &outputs[index] : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().name == "input.1");
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 64, 64});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensorMap outputs{};
            outputs.reserve(9U);
            const std::int64_t rows[] = {128, 32, 8};
            for (std::size_t level = 0; level < 3U; ++level)
            {
                cvkit::infer::detail::RawTensor score{};
                score.name  = "score";
                score.shape = {1, rows[level], 1};
                score.data.assign(static_cast<std::size_t>(rows[level]), 0.01F);
                if (level == 0)
                {
                    score.data[18] = 0.95F;
                }
                outputs.push_back(std::move(score));
            }
            for (std::size_t level = 0; level < 3U; ++level)
            {
                cvkit::infer::detail::RawTensor bbox{};
                bbox.name  = "bbox";
                bbox.shape = {1, rows[level], 4};
                bbox.data.assign(static_cast<std::size_t>(rows[level] * 4), 0.0F);
                if (level == 0)
                {
                    const auto offset = 18U * 4U;
                    bbox.data[offset + 0U] = 1.0F;
                    bbox.data[offset + 1U] = 1.0F;
                    bbox.data[offset + 2U] = 1.5F;
                    bbox.data[offset + 3U] = 2.0F;
                }
                outputs.push_back(std::move(bbox));
            }
            for (std::size_t level = 0; level < 3U; ++level)
            {
                cvkit::infer::detail::RawTensor kps{};
                kps.name  = "kps";
                kps.shape = {1, rows[level], 10};
                kps.data.assign(static_cast<std::size_t>(rows[level] * 10), 0.0F);
                if (level == 0)
                {
                    const auto offset = 18U * 10U;
                    kps.data[offset + 0U] = -0.5F;
                    kps.data[offset + 1U] = -0.5F;
                    kps.data[offset + 2U] = 0.5F;
                    kps.data[offset + 3U] = -0.5F;
                    kps.data[offset + 4U] = 0.0F;
                    kps.data[offset + 5U] = 0.0F;
                    kps.data[offset + 6U] = -0.5F;
                    kps.data[offset + 7U] = 1.0F;
                    kps.data[offset + 8U] = 0.5F;
                    kps.data[offset + 9U] = 1.0F;
                }
                outputs.push_back(std::move(kps));
            }
            return outputs;
        }
    };

    class StubPoseBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "images",
                {1, 3, 8, 8},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo output{
                "keypoints",
                {1, 3, 3},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &output : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 8, 8});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensor output{};
            output.name  = "keypoints";
            output.shape = {1, 3, 3};
            output.data  = {
                1.0F, 2.0F, 0.9F,
                3.0F, 4.0F, 0.8F,
                5.0F, 6.0F, 0.7F};
            return {std::move(output)};
        }
    };

    class StubFaceMeshBackendSession final : public cvkit::infer::detail::IBackendSession
    {
      public:
        bool load(const cvkit::infer::ModelSpec& spec) override
        {
            static_cast<void>(spec);
            return true;
        }

        [[nodiscard]] bool ready() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::Backend backend() const override
        {
            return cvkit::infer::Backend::onnxruntime;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* input_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo input{
                "images",
                {1, 3, 16, 16},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &input : nullptr;
        }

        [[nodiscard]] const cvkit::infer::TensorInfo* output_info(std::size_t index = 0) const override
        {
            static const cvkit::infer::TensorInfo output{
                "landmarks",
                {1, 2, 4},
                cvkit::infer::TensorDataType::float32,
                cvkit::infer::MemoryDevice::host};
            return index == 0 ? &output : nullptr;
        }

        [[nodiscard]] cvkit::infer::detail::RawTensorMap run(
            const cvkit::infer::detail::RawTensorMap& inputs) const override
        {
            REQUIRE(inputs.size() == 1);
            REQUIRE(inputs.front().shape == std::vector<std::int64_t>{1, 3, 16, 16});
            REQUIRE(inputs.front().data_type == cvkit::infer::TensorDataType::float32);
            REQUIRE(inputs.front().memory_device == cvkit::infer::MemoryDevice::host);

            cvkit::infer::detail::RawTensor output{};
            output.name  = "landmarks";
            output.shape = {1, 2, 4};
            output.data  = {
                1.0F, 2.0F, 0.0F, 0.95F,
                3.0F, 4.0F, 0.1F, 0.85F};
            return {std::move(output)};
        }
    };

    class AsyncTestNode final : public cvkit::infer::detail::INode
    {
      public:
        explicit AsyncTestNode(std::string_view node_name)
            : name_(node_name)
        {
        }

        [[nodiscard]] std::string_view name() const override
        {
            return name_;
        }

        [[nodiscard]] std::vector<std::string> consumes() const override
        {
            return {"input:image"};
        }

        [[nodiscard]] std::vector<std::string> produces() const override
        {
            return {"output:async-node"};
        }

        [[nodiscard]] bool supports_async() const override
        {
            return true;
        }

        [[nodiscard]] cvkit::infer::detail::PacketFuture submit(cvkit::infer::detail::Packet packet) const override
        {
            auto future = std::async(
                              std::launch::async,
                              [name = name_, packet = std::move(packet)]() mutable
                              {
                                  packet.output.add("node", name);
                                  packet.put(std::string{name}, 1);
                                  return packet;
                              })
                              .share();
            return cvkit::infer::detail::PacketFuture{std::move(future)};
        }

        [[nodiscard]] cvkit::infer::detail::Packet process(cvkit::infer::detail::Packet packet) const override
        {
            packet.output.add("node", std::string{name_});
            packet.put(std::string{name_}, 1);
            return packet;
        }

      private:
        std::string name_{};
    };

    class ErrorTraceNode final : public cvkit::infer::detail::INode
    {
      public:
        explicit ErrorTraceNode(std::string_view node_name)
            : name_(node_name)
        {
        }

        [[nodiscard]] std::string_view name() const override
        {
            return name_;
        }

        [[nodiscard]] cvkit::infer::detail::Packet process(cvkit::infer::detail::Packet packet) const override
        {
            packet.put("node.error", std::string{"cuda_device preprocess path is not implemented yet"});
            return packet;
        }

        [[nodiscard]] std::string trace_message(const cvkit::infer::detail::Packet& packet) const override
        {
            if (const auto* error = packet.get<std::string>("node.error"); error != nullptr)
            {
                return *error;
            }
            return {};
        }

      private:
        std::string name_{};
    };

    void check_face_tiled_trace(const std::vector<cvkit::infer::GraphTraceInfo>& trace)
    {
        REQUIRE(trace.size() == 13);
        CHECK(trace.front().name == "tiled_inference");
        CHECK(trace.front().sequence == 0);
        CHECK(trace.front().scratch_count == 12);
        CHECK(trace.front().message.find("tiles=12") != std::string::npos);
        CHECK(trace.front().message.find("source=2048x1150") != std::string::npos);
        CHECK(trace.front().message.find("tile=640x640") != std::string::npos);
        REQUIRE(trace.front().has_tile_info);
        CHECK(trace.front().tile_info.aggregate);
        CHECK(trace.front().tile_info.tile_count == 12);
        CHECK(trace.front().tile_info.source_width == 2048);
        CHECK(trace.front().tile_info.source_height == 1150);
        CHECK(trace.front().tile_info.tile_width == 640);
        CHECK(trace.front().tile_info.tile_height == 640);
        CHECK(trace.front().tile_info.overlap_x == 160);
        CHECK(trace.front().tile_info.overlap_y == 160);
        CHECK(trace[1].name == "tile_0");
        CHECK(trace[1].message.find("x=0 y=0 width=640 height=640") != std::string::npos);
        REQUIRE(trace[1].has_tile_info);
        CHECK_FALSE(trace[1].tile_info.aggregate);
        CHECK(trace[1].tile_info.tile_index == 0);
        CHECK(trace[1].tile_info.x == 0);
        CHECK(trace[1].tile_info.y == 0);
        CHECK(trace[1].tile_info.width == 640);
        CHECK(trace[1].tile_info.height == 640);
        CHECK(trace.back().name == "tile_11");
        CHECK(trace.back().message.find("tile_index=11") != std::string::npos);
        REQUIRE(trace.back().has_tile_info);
        CHECK(trace.back().tile_info.tile_index == 11);
        CHECK(trace.back().tile_info.x == 1408);
        CHECK(trace.back().tile_info.y == 510);
        CHECK(trace.back().tile_info.width == 640);
        CHECK(trace.back().tile_info.height == 640);
    }

    void check_face_tiled_json(const cvkit::infer::Model& model, bool async_infer)
    {
        const auto json = cvkit::infer::build_graph_json(model, async_infer);
        CHECK(json.find("\"tiling\": {") != std::string::npos);
        CHECK(json.find("\"tile_count\": 12") != std::string::npos);
        CHECK(json.find("\"source_width\": 2048") != std::string::npos);
        CHECK(json.find("\"source_height\": 1150") != std::string::npos);
        CHECK(json.find("\"tile_width\": 640") != std::string::npos);
        CHECK(json.find("\"overlap_x\": 160") != std::string::npos);
        CHECK(json.find("\"tile_index\": 11") != std::string::npos);
        CHECK(json.find("\"tile_info\": {\"aggregate\":true") != std::string::npos);
        CHECK(json.find("\"tile_info\": {\"aggregate\":false") != std::string::npos);
    }

}  // namespace

TEST_CASE("model load fails for missing onnx file")
{
    cvkit::infer::Model     model;
    cvkit::infer::ModelSpec spec{};
    spec.model_path = "missing.onnx";
    spec.backend    = cvkit::infer::Backend::none;
    spec.task       = cvkit::infer::TaskKind::detection;
    spec.family     = "yolo11";
    CHECK_FALSE(model.load(spec));
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

TEST_CASE("model exposes configurable tile options")
{
    cvkit::infer::Model model;
    CHECK_FALSE(model.tile_options().enabled);

    cvkit::infer::TileOptions options{};
    options.enabled     = true;
    options.tile_width  = 320;
    options.tile_height = 240;
    options.overlap_x   = 32;
    options.overlap_y   = 24;
    model.set_tile_options(options);

    CHECK(model.tile_options().enabled);
    CHECK(model.tile_options().tile_width == 320);
    CHECK(model.tile_options().tile_height == 240);
    CHECK(model.tile_options().overlap_x == 32);
    CHECK(model.tile_options().overlap_y == 24);
}

TEST_CASE("model session_info exposes backend tensor metadata for yolo model")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";

    REQUIRE(std::filesystem::exists(model_path));

    cvkit::infer::Backend backend = cvkit::infer::Backend::none;
#if defined(CVKIT_WITH_ONNXRUNTIME)
    backend = cvkit::infer::Backend::onnxruntime;
#elif defined(CVKIT_WITH_TENSORRT)
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to validate TensorRT session metadata");
    }
    backend = cvkit::infer::Backend::tensorrt;
#else
    SKIP("no inference backend is enabled in this build");
#endif

    cvkit::infer::ModelSpec spec{};
    spec.model_path = model_path.string();
    spec.backend    = backend;
    spec.task       = cvkit::infer::TaskKind::detection;
    spec.family     = "yolo11";

    cvkit::infer::Model model;
    REQUIRE(model.load(spec));

    const auto info = model.session_info();
    REQUIRE_FALSE(info.inputs.empty());
    REQUIRE_FALSE(info.outputs.empty());
    CHECK_FALSE(info.inputs.front().shape.empty());
    CHECK_FALSE(info.outputs.front().shape.empty());
}

TEST_CASE("tensor file round-trip preserves tensor metadata and values")
{
    const auto                temp_dir    = std::filesystem::temp_directory_path() / "cvkit_tensor_io";
    const auto                tensor_path = temp_dir / "tensor.bin";

    cvkit::infer::TensorValue saved{};
    saved.name          = "image_embeddings";
    saved.shape         = {1, 256, 64, 64};
    saved.data          = {0.25F, -0.5F, 1.0F, 2.5F};
    saved.data_type     = cvkit::infer::TensorDataType::float32;
    saved.memory_device = cvkit::infer::MemoryDevice::host;

    REQUIRE(cvkit::infer::save_tensor_file(saved, tensor_path));

    cvkit::infer::TensorValue loaded{};
    REQUIRE(cvkit::infer::load_tensor_file(tensor_path, loaded));
    CHECK(loaded.name == saved.name);
    CHECK(loaded.shape == saved.shape);
    CHECK(loaded.data == saved.data);
    CHECK(loaded.data_type == saved.data_type);
    CHECK(loaded.memory_device == saved.memory_device);
    CHECK(loaded.storage == saved.storage);
    CHECK(loaded.is_host_accessible());
}

TEST_CASE("task input and output support richer infer value types")
{
    cvkit::core::Frame image_frame{};
    image_frame.desc.width    = 4;
    image_frame.desc.height   = 3;
    image_frame.desc.channels = 3;
    image_frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    image_frame.data.assign(static_cast<std::size_t>(4 * 3 * 3), static_cast<std::uint8_t>(7));

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{image_frame});
    input.add("boxes", cvkit::infer::BoxListValue{{cvkit::core::BBox{1.0F, 2.0F, 3.0F, 4.0F}}});
    input.add("keypoints", cvkit::infer::KeypointsValue{{cvkit::core::Point2f{5.0F, 6.0F}}});

    const auto* image     = input.find<cvkit::infer::ImageValue>("image");
    const auto* boxes     = input.find<cvkit::infer::BoxListValue>("boxes");
    const auto* keypoints = input.find<cvkit::infer::KeypointsValue>("keypoints");
    REQUIRE(image != nullptr);
    REQUIRE(boxes != nullptr);
    REQUIRE(keypoints != nullptr);
    CHECK(image->memory_device == cvkit::infer::MemoryDevice::host);
    CHECK(image->storage == cvkit::infer::StorageKind::owned);
    CHECK(image->owns_storage());
    CHECK(image->is_host_accessible());
    CHECK(image->bytes_per_pixel() == 3);
    CHECK(image->packed_row_stride_bytes() == 12);
    CHECK(image->effective_row_stride_bytes() == 12);
    CHECK(image->is_packed());
    CHECK(image->has_valid_host_layout());

    auto padded_image             = *image;
    padded_image.row_stride_bytes = 16;
    CHECK(padded_image.effective_row_stride_bytes() == 16);
    CHECK_FALSE(padded_image.is_packed());
    CHECK_FALSE(padded_image.has_valid_host_layout());
    padded_image.frame.data.resize(16U * 3U);
    CHECK(padded_image.has_valid_host_layout());
    CHECK_FALSE(image->has_valid_device_view());

    cvkit::infer::ImageValue nv12_image{};
    nv12_image.frame.desc.width    = 8;
    nv12_image.frame.desc.height   = 4;
    nv12_image.frame.desc.channels = 1;
    nv12_image.frame.desc.format   = cvkit::core::PixelFormat::nv12;
    nv12_image.row_stride_bytes    = 8;
    nv12_image.frame.data.assign(8U * 4U * 3U / 2U, static_cast<std::uint8_t>(128));
    CHECK(nv12_image.bytes_per_pixel() == 1);
    CHECK(nv12_image.packed_row_stride_bytes() == 8);
    CHECK(nv12_image.required_byte_size() == 48);
    CHECK(nv12_image.has_valid_host_layout());

    auto short_nv12 = nv12_image;
    short_nv12.frame.data.resize(8U * 4U);
    CHECK_FALSE(short_nv12.has_valid_host_layout());

    cvkit::infer::ImageValue cuda_nv12 = nv12_image;
    cuda_nv12.memory_device            = cvkit::infer::MemoryDevice::cuda;
    cuda_nv12.device                   = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    cuda_nv12.storage                  = cvkit::infer::StorageKind::external_view;
    cuda_nv12.external_data            = reinterpret_cast<const void*>(0x1);
    cuda_nv12.storage_bytes            = nv12_image.required_byte_size();
    cuda_nv12.frame.data.clear();
    CHECK(cuda_nv12.has_valid_device_view());
    cuda_nv12.storage_bytes = 8U * 4U;
    CHECK_FALSE(cuda_nv12.has_valid_device_view());

    REQUIRE(boxes->boxes.size() == 1);
    REQUIRE(keypoints->points.size() == 1);

    cvkit::infer::ClassificationValue classification{};
    classification.class_id = 1;
    classification.score    = 0.9F;
    classification.label    = "dog";
    input.add("classification", classification);
    const auto* stored_classification = input.find<cvkit::infer::ClassificationValue>("classification");
    REQUIRE(stored_classification != nullptr);
    CHECK(stored_classification->class_id == 1);
    CHECK(stored_classification->score == Catch::Approx(0.9F));
    CHECK(stored_classification->label == "dog");

    cvkit::infer::TensorValue tensor{};
    tensor.name  = "host_tensor";
    tensor.shape = {1, 4};
    tensor.data  = {1.0F, 2.0F, 3.0F, 4.0F};
    CHECK(tensor.data_type == cvkit::infer::TensorDataType::float32);
    CHECK(tensor.memory_device == cvkit::infer::MemoryDevice::host);
    CHECK(tensor.storage == cvkit::infer::StorageKind::owned);
    CHECK(tensor.owns_storage());
    CHECK(tensor.is_host_accessible());
    CHECK(tensor.element_count() == 4);
    CHECK(tensor.packed_byte_size() == sizeof(float) * 4U);
    CHECK(tensor.byte_size() == sizeof(float) * 4U);
    CHECK(tensor.is_packed());
    CHECK(tensor.has_valid_host_layout());

    tensor.data.push_back(5.0F);
    CHECK(tensor.byte_size() == sizeof(float) * 5U);
    CHECK_FALSE(tensor.is_packed());
    CHECK(tensor.has_valid_host_layout());

    tensor.storage = cvkit::infer::StorageKind::external_view;
    CHECK_FALSE(tensor.owns_storage());
    CHECK_FALSE(tensor.has_valid_device_view());

    cvkit::infer::TensorValue device_tensor{};
    device_tensor.name          = "device_tensor";
    device_tensor.shape         = {1, 4};
    device_tensor.data_type     = cvkit::infer::TensorDataType::float32;
    device_tensor.memory_device = cvkit::infer::MemoryDevice::cuda;
    device_tensor.storage       = cvkit::infer::StorageKind::external_view;
    device_tensor.external_data = reinterpret_cast<const void*>(0x1);
    device_tensor.storage_bytes = sizeof(float) * 4U;
    CHECK(device_tensor.has_valid_device_view());
    CHECK_FALSE(device_tensor.has_valid_host_layout());

    cvkit::infer::ImageValue device_image = *image;
    device_image.memory_device            = cvkit::infer::MemoryDevice::cuda;
    device_image.storage                  = cvkit::infer::StorageKind::external_view;
    device_image.external_data            = reinterpret_cast<const void*>(0x1);
    device_image.storage_bytes            = image->frame.data.size();
    device_image.frame.data.clear();
    CHECK(device_image.has_valid_device_view());
    CHECK_FALSE(device_image.has_valid_host_layout());

    cvkit::infer::TaskOutput output{};
    cvkit::core::Frame       mask_frame{};
    mask_frame.desc.width    = 8;
    mask_frame.desc.height   = 6;
    mask_frame.desc.channels = 1;
    mask_frame.data.assign(static_cast<std::size_t>(8 * 6), static_cast<std::uint8_t>(255));
    output.add("mask", cvkit::infer::MaskValue{mask_frame});

    const auto* mask = output.find<cvkit::infer::MaskValue>("mask");
    REQUIRE(mask != nullptr);
    CHECK(mask->frame.desc.width == 8);
    CHECK(mask->frame.desc.height == 6);
    CHECK(mask->frame.desc.channels == 1);
    CHECK(mask->frame.data.size() == 48);
}

TEST_CASE("classification pipeline returns top1 label and scores")
{
    StubClassificationBackendSession             backend{};
    cvkit::infer::detail::ClassificationPipeline pipeline{};

    cvkit::infer::detail::PipelineContext        context{};
    context.spec.task   = cvkit::infer::TaskKind::classification;
    context.spec.family = "classification";
    context.labels      = {"cat", "dog", "bird"};

    cvkit::core::Frame frame{};
    frame.desc.width    = 8;
    frame.desc.height   = 8;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(8 * 8 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output         = pipeline.run_sync(backend, input, context);
    const auto* classification = output.find<cvkit::infer::ClassificationValue>("classification");
    const auto* scores         = output.find<std::vector<float>>("scores");
    REQUIRE(classification != nullptr);
    REQUIRE(scores != nullptr);
    CHECK(classification->class_id == 1);
    CHECK(classification->score == Catch::Approx(0.85F));
    CHECK(classification->label == "dog");
    CHECK(scores->size() == 3);
    CHECK(scores->at(1) == Catch::Approx(0.85F));
}

TEST_CASE("segmentation pipeline returns mask and logits")
{
    StubSegmentationBackendSession             backend{};
    cvkit::infer::detail::SegmentationPipeline pipeline{};

    cvkit::infer::detail::PipelineContext context{};
    context.spec.task   = cvkit::infer::TaskKind::segmentation;
    context.spec.family = "segmentation";

    cvkit::core::Frame frame{};
    frame.desc.width    = 4;
    frame.desc.height   = 4;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(4 * 4 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output = pipeline.run_sync(backend, input, context);
    const auto* mask   = output.find<cvkit::infer::MaskValue>("mask");
    const auto* logits = output.find<cvkit::infer::TensorValue>("logits");
    REQUIRE(mask != nullptr);
    REQUIRE(logits != nullptr);
    CHECK(mask->frame.desc.width == 4);
    CHECK(mask->frame.desc.height == 4);
    CHECK(mask->frame.desc.channels == 1);
    REQUIRE(mask->frame.data.size() == 16);
    CHECK(mask->frame.data[0] == 1);
    CHECK(mask->frame.data[1] == 0);
    CHECK(logits->shape == std::vector<std::int64_t>{1, 2, 4, 4});
}

TEST_CASE("face detection pipeline returns bbox and five landmarks")
{
    StubFaceDetectionBackendSession             backend{};
    cvkit::infer::detail::FaceDetectionPipeline pipeline{};

    cvkit::infer::detail::PipelineContext context{};
    context.spec.task            = cvkit::infer::TaskKind::face_detection;
    context.spec.family          = "decoded_rows";
    context.confidence_threshold = 0.6F;
    context.iou_threshold        = 0.4F;

    cvkit::core::Frame frame{};
    frame.desc.width    = 64;
    frame.desc.height   = 64;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(64 * 64 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output     = pipeline.run_sync(backend, input, context);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    REQUIRE(detections->size() == 1);
    CHECK(detections->front().class_id == 0);
    CHECK(detections->front().score == Catch::Approx(0.95F));
    CHECK(detections->front().box.x == Catch::Approx(10.0F));
    CHECK(detections->front().box.y == Catch::Approx(12.0F));
    CHECK(detections->front().box.width == Catch::Approx(20.0F));
    CHECK(detections->front().box.height == Catch::Approx(24.0F));
    REQUIRE(detections->front().keypoints.size() == 5);
    CHECK(detections->front().keypoints[0].x == Catch::Approx(14.0F));
    CHECK(detections->front().keypoints[0].y == Catch::Approx(18.0F));
}

TEST_CASE("face detection pipeline decodes SCRFD stride outputs")
{
    StubScrfdBackendSession                       backend{};
    cvkit::infer::detail::FaceDetectionPipeline   pipeline{};

    cvkit::infer::detail::PipelineContext context{};
    context.spec.task            = cvkit::infer::TaskKind::face_detection;
    context.spec.family          = "scrfd";
    context.confidence_threshold = 0.6F;
    context.iou_threshold        = 0.4F;

    cvkit::core::Frame frame{};
    frame.desc.width    = 64;
    frame.desc.height   = 64;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(64 * 64 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output     = pipeline.run_sync(backend, input, context);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    REQUIRE(detections->size() == 1);
    const auto& face = detections->front();
    CHECK(face.score == Catch::Approx(0.95F));
    CHECK(face.box.x == Catch::Approx(0.0F));
    CHECK(face.box.y == Catch::Approx(0.0F));
    CHECK(face.box.width == Catch::Approx(20.0F));
    CHECK(face.box.height == Catch::Approx(24.0F));
    REQUIRE(face.keypoints.size() == 5);
    CHECK(face.keypoints[0].x == Catch::Approx(4.0F));
    CHECK(face.keypoints[0].y == Catch::Approx(4.0F));
}

TEST_CASE("face detection runs SCRFD ONNX model and returns five-point landmarks")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "scrfd_10g.onnx";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(model_path))
    {
        SKIP("scrfd_10g.onnx is not present under assets/models");
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path = model_path.string();
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::face_detection;
    spec.family     = "scrfd";

    cvkit::infer::Model model;
    model.set_confidence_threshold(0.02F);
    model.set_iou_threshold(0.4F);
    if (!model.load(spec))
    {
        SKIP("onnxruntime face detection backend is not available in this build");
    }

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
    const auto  output     = model.run_sync(input);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    REQUIRE_FALSE(detections->empty());
    CHECK(std::any_of(
        detections->begin(),
        detections->end(),
        [](const cvkit::core::Detection& detection)
        { return detection.keypoints.size() == 5; }));
}

TEST_CASE("face detection tiled model merges detections back to source coordinates")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "scrfd_10g_ac133ba7.onnx";
    const auto image_path  = source_root / "assets" / "images" / "face.jpg";

    if (!std::filesystem::exists(model_path) || !std::filesystem::exists(image_path))
    {
        SKIP("scrfd_10g_ac133ba7.onnx or face.jpg is not present under assets");
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path = model_path.string();
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::face_detection;
    spec.family     = "scrfd_raw_bgr";

    cvkit::infer::Model model;
    model.set_confidence_threshold(0.5F);
    model.set_iou_threshold(0.4F);
    if (!model.load(spec))
    {
        SKIP("onnxruntime face detection backend is not available in this build");
    }

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

    const auto  full_output     = model.run_sync(input);
    const auto* full_detections = full_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(full_detections != nullptr);
    REQUIRE_FALSE(full_detections->empty());

    cvkit::infer::TileOptions tile_options{};
    tile_options.enabled     = true;
    tile_options.tile_width  = 640;
    tile_options.tile_height = 640;
    tile_options.overlap_x   = 160;
    tile_options.overlap_y   = 160;
    model.set_tile_options(tile_options);

    const auto  tiled_output     = model.run_sync(input);
    const auto* tiled_detections = tiled_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(tiled_detections != nullptr);
    CHECK(tiled_detections->size() >= full_detections->size());
    CHECK(std::all_of(
        tiled_detections->begin(),
        tiled_detections->end(),
        [&frame](const cvkit::core::Detection& detection)
        {
            return detection.box.x >= 0.0F &&
                   detection.box.y >= 0.0F &&
                   detection.box.x <= static_cast<float>(frame.desc.width) &&
                   detection.box.y <= static_cast<float>(frame.desc.height);
        }));

    const auto trace = model.last_graph_trace();
    check_face_tiled_trace(trace);
    check_face_tiled_json(model, false);
}

TEST_CASE("face detection tiled async model stores tile trace")
{
    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "scrfd_10g_ac133ba7.onnx";
    const auto image_path  = source_root / "assets" / "images" / "face.jpg";

    if (!std::filesystem::exists(model_path) || !std::filesystem::exists(image_path))
    {
        SKIP("scrfd_10g_ac133ba7.onnx or face.jpg is not present under assets");
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path = model_path.string();
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::face_detection;
    spec.family     = "scrfd_raw_bgr";

    cvkit::infer::Model model;
    model.set_confidence_threshold(0.5F);
    model.set_iou_threshold(0.4F);
    if (!model.load(spec))
    {
        SKIP("onnxruntime face detection backend is not available in this build");
    }

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

    cvkit::infer::TileOptions tile_options{};
    tile_options.enabled     = true;
    tile_options.tile_width  = 640;
    tile_options.tile_height = 640;
    tile_options.overlap_x   = 160;
    tile_options.overlap_y   = 160;
    model.set_tile_options(tile_options);

    auto future = model.submit(input);
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);

    const auto  output     = future.get();
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    CHECK(detections->size() > 500U);

    const auto trace = model.last_graph_trace();
    check_face_tiled_trace(trace);
    check_face_tiled_json(model, true);
}

TEST_CASE("pose pipeline returns keypoints and scores")
{
    StubPoseBackendSession             backend{};
    cvkit::infer::detail::PosePipeline pipeline{};

    cvkit::infer::detail::PipelineContext context{};
    context.spec.task   = cvkit::infer::TaskKind::pose;
    context.spec.family = "pose";

    cvkit::core::Frame frame{};
    frame.desc.width    = 8;
    frame.desc.height   = 8;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(8 * 8 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output    = pipeline.run_sync(backend, input, context);
    const auto* keypoints = output.find<cvkit::infer::KeypointsValue>("keypoints");
    const auto* scores    = output.find<std::vector<float>>("scores");
    const auto* raw       = output.find<cvkit::infer::TensorValue>("raw");
    REQUIRE(keypoints != nullptr);
    REQUIRE(scores != nullptr);
    REQUIRE(raw != nullptr);
    REQUIRE(keypoints->points.size() == 3);
    CHECK(keypoints->points[0].x == Catch::Approx(1.0F));
    CHECK(keypoints->points[0].y == Catch::Approx(2.0F));
    CHECK(scores->size() == 3);
    CHECK(scores->at(1) == Catch::Approx(0.8F));
    CHECK(raw->shape == std::vector<std::int64_t>{1, 3, 3});
}

TEST_CASE("facemesh pipeline returns landmarks and scores")
{
    StubFaceMeshBackendSession             backend{};
    cvkit::infer::detail::FaceMeshPipeline pipeline{};

    cvkit::infer::detail::PipelineContext context{};
    context.spec.task   = cvkit::infer::TaskKind::facemesh;
    context.spec.family = "facemesh";

    cvkit::core::Frame frame{};
    frame.desc.width    = 16;
    frame.desc.height   = 16;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(16 * 16 * 3), 127U);

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});

    const auto  output    = pipeline.run_sync(backend, input, context);
    const auto* landmarks = output.find<cvkit::infer::KeypointsValue>("landmarks");
    const auto* scores    = output.find<std::vector<float>>("scores");
    const auto* raw       = output.find<cvkit::infer::TensorValue>("raw");
    REQUIRE(landmarks != nullptr);
    REQUIRE(scores != nullptr);
    REQUIRE(raw != nullptr);
    REQUIRE(landmarks->points.size() == 2);
    CHECK(landmarks->points[0].x == Catch::Approx(1.0F));
    CHECK(landmarks->points[0].y == Catch::Approx(2.0F));
    CHECK(scores->size() == 2);
    CHECK(scores->at(1) == Catch::Approx(0.85F));
    CHECK(raw->shape == std::vector<std::int64_t>{1, 2, 4});
}

TEST_CASE("backend input tensor contract currently requires host float32 tensors")
{
    cvkit::infer::detail::RawTensor tensor{};
    tensor.name  = "input";
    tensor.shape = {1, 3, 4, 4};
    tensor.data  = std::vector<float>(48, 1.0F);

    CHECK(cvkit::infer::detail::is_supported_backend_input_tensor(tensor));

    tensor.data_type = cvkit::infer::TensorDataType::float16;
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_input_tensor(tensor));

    tensor.data_type     = cvkit::infer::TensorDataType::float32;
    tensor.memory_device = cvkit::infer::MemoryDevice::cuda;
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_input_tensor(tensor));

    tensor.memory_device = cvkit::infer::MemoryDevice::host;
    tensor.shape.clear();
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_input_tensor(tensor));

    tensor.shape = {1, 3, 4, 4};
    tensor.data.assign(16, 1.0F);
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_input_tensor(tensor));
}

TEST_CASE("yolo preprocess source selection distinguishes host and cuda image inputs")
{
    cvkit::core::Frame frame{};
    frame.desc.width    = 4;
    frame.desc.height   = 3;
    frame.desc.channels = 3;
    frame.data.assign(static_cast<std::size_t>(4 * 3 * 3), static_cast<std::uint8_t>(9));

    cvkit::infer::TaskInput host_input{};
    host_input.add("image", cvkit::infer::ImageValue{frame});
    const auto host_source = cvkit::infer::detail::select_yolo_preprocess_source(host_input);
    CHECK(host_source.path == cvkit::infer::detail::YoloPreprocessPath::host_cpu);
    REQUIRE(host_source.frame != nullptr);
    CHECK(host_source.memory_device == cvkit::infer::MemoryDevice::host);

    cvkit::infer::ImageValue cuda_image{};
    cuda_image.frame         = frame;
    cuda_image.memory_device = cvkit::infer::MemoryDevice::cuda;
    cuda_image.device        = {cvkit::infer::DeviceKind::cuda, 0};
    cuda_image.storage       = cvkit::infer::StorageKind::external_view;
    cuda_image.external_data = reinterpret_cast<const void*>(0x1);
    cuda_image.storage_bytes = static_cast<std::size_t>(4 * 3 * 3);

    cvkit::infer::TaskInput cuda_input{};
    cuda_input.add("image", cuda_image);
    const auto cuda_source = cvkit::infer::detail::select_yolo_preprocess_source(cuda_input);
    CHECK(cuda_source.path == cvkit::infer::detail::YoloPreprocessPath::cuda_device);
    CHECK(cuda_source.frame == nullptr);
    CHECK(cuda_source.image != nullptr);
    CHECK(cuda_source.memory_device == cvkit::infer::MemoryDevice::cuda);
    CHECK(cuda_source.device.kind == cvkit::infer::DeviceKind::cuda);
}

TEST_CASE("yolo cuda preprocess accepts nv12 device frame input")
{
#if !defined(CVKIT_WITH_CUDA_RUNTIME) || !defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    SKIP("CUDA runtime and preprocess kernel are required for NV12 device-frame preprocess");
#else
    const int width = 64;
    const int height = 48;
    const auto stride = static_cast<std::size_t>(width);
    const auto bytes = stride * static_cast<std::size_t>(height) * 3U / 2U;

    void* device_ptr = nullptr;
    REQUIRE(cudaMalloc(&device_ptr, bytes) == cudaSuccess);
    REQUIRE(cudaMemset(device_ptr, 128, bytes) == cudaSuccess);

    cvkit::core::DeviceFrame device_frame{};
    device_frame.desc.width = width;
    device_frame.desc.height = height;
    device_frame.desc.channels = 1;
    device_frame.desc.format = cvkit::core::PixelFormat::nv12;
    device_frame.data = reinterpret_cast<std::uintptr_t>(device_ptr);
    device_frame.bytes = bytes;
    device_frame.stride_bytes = stride;
    device_frame.memory_device = cvkit::core::MemoryDevice::cuda;
    device_frame.device_index = 0;
    device_frame.owner = std::shared_ptr<void>(
        device_ptr,
        [](void* ptr)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        });
    REQUIRE(device_frame.valid());

    const auto image = cvkit::infer::image_value_from_device_frame(device_frame);
    REQUIRE(image.has_valid_device_view());
    CHECK(image.storage_owner == device_frame.owner);

    cvkit::infer::TaskInput input{};
    input.add("image", image);

    const auto outcome = cvkit::infer::detail::preprocess_yolo(input, {1, 3, 32, 32}, true);
    REQUIRE(outcome.ok());
    CHECK(outcome.result.tensor.memory_device == cvkit::infer::MemoryDevice::cuda);
    CHECK(outcome.result.tensor.storage == cvkit::infer::StorageKind::owned);
    CHECK(outcome.result.tensor.shape == std::vector<std::int64_t>{1, 3, 32, 32});
    CHECK(outcome.result.tensor.has_valid_device_view());
#endif
}

TEST_CASE("promptable cuda preprocess accepts nv12 device image")
{
#if !defined(CVKIT_WITH_CUDA_RUNTIME) || !defined(CVKIT_WITH_CUDA_PREPROCESS_KERNEL)
    SKIP("CUDA runtime and preprocess kernel are required for NV12 promptable preprocess");
#else
    const int width = 64;
    const int height = 48;
    const auto stride = static_cast<std::size_t>(width);
    const auto bytes = stride * static_cast<std::size_t>(height) * 3U / 2U;

    void* device_ptr = nullptr;
    REQUIRE(cudaMalloc(&device_ptr, bytes) == cudaSuccess);
    REQUIRE(cudaMemset(device_ptr, 128, bytes) == cudaSuccess);

    cvkit::infer::ImageValue image{};
    image.frame.desc.width    = width;
    image.frame.desc.height   = height;
    image.frame.desc.channels = 1;
    image.frame.desc.format   = cvkit::core::PixelFormat::nv12;
    image.memory_device       = cvkit::infer::MemoryDevice::cuda;
    image.device              = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    image.storage             = cvkit::infer::StorageKind::external_view;
    image.row_stride_bytes    = stride;
    image.external_data       = device_ptr;
    image.storage_bytes       = bytes;
    image.storage_owner       = std::shared_ptr<void>(
        device_ptr,
        [](void* ptr)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        });
    REQUIRE(image.has_valid_device_view());

    std::string error;
    auto        tensor = cvkit::infer::detail::preprocess_promptable_encoder_cuda(image, true, &error);
    REQUIRE(tensor.has_value());
    CHECK(error.empty());
    CHECK(tensor->name == "batched_images");
    CHECK(tensor->shape == std::vector<std::int64_t>{1, 3, 1024, 1024});
    CHECK(tensor->memory_device == cvkit::infer::MemoryDevice::cuda);
    CHECK(tensor->has_valid_device_view());
#endif
}

TEST_CASE("backend output tensor contract currently supports float32 export only")
{
    CHECK(cvkit::infer::detail::is_supported_backend_output_tensor_type(
        cvkit::infer::TensorDataType::float32));

    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_output_tensor_type(
        cvkit::infer::TensorDataType::float16));
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_output_tensor_type(
        cvkit::infer::TensorDataType::int32));
    CHECK_FALSE(cvkit::infer::detail::is_supported_backend_output_tensor_type(
        cvkit::infer::TensorDataType::unknown));
}

TEST_CASE("task graph records node metadata and timing trace")
{
    cvkit::infer::detail::TaskGraph graph{};
    graph.add_node(std::make_shared<TestNode>("first"));
    graph.add_node(std::make_shared<TestNode>("second"));

    const auto metadata = graph.metadata();
    REQUIRE(metadata.size() == 2);
    CHECK(metadata[0].name == "first");
    CHECK(metadata[1].name == "second");
    CHECK(metadata[0].depends_on.empty());
    CHECK(metadata[1].depends_on == std::vector<std::string>{"first"});

    const auto boundary = graph.boundary();
    CHECK(boundary.inputs.empty());
    CHECK(boundary.outputs.empty());

    cvkit::infer::detail::Packet packet{};
    packet.input.add("image", cvkit::core::Frame{});
    const auto result = graph.run_sync(std::move(packet));

    REQUIRE(result.trace.size() == 2);
    CHECK(result.trace[0].name == "first");
    CHECK(result.trace[0].sequence == 0);
    CHECK(result.trace[1].name == "second");
    CHECK(result.trace[1].sequence == 1);
    CHECK(result.trace[0].duration_us >= 0);
    CHECK(result.trace[1].duration_us >= 0);
}

TEST_CASE("task graph honors explicit node dependencies")
{
    cvkit::infer::detail::TaskGraph graph{};
    graph.add_node(std::make_shared<TestNode>("final"), {"middle"});
    graph.add_node(std::make_shared<TestNode>("start"), std::vector<std::string>{});
    graph.add_node(std::make_shared<TestNode>("middle"), {"start"});

    const auto metadata = graph.metadata();
    REQUIRE(metadata.size() == 3);
    CHECK(metadata[0].name == "final");
    CHECK(metadata[0].depends_on == std::vector<std::string>{"middle"});
    CHECK(metadata[2].name == "middle");
    CHECK(metadata[2].depends_on == std::vector<std::string>{"start"});

    cvkit::infer::detail::Packet packet{};
    const auto                   result = graph.run_sync(std::move(packet));

    REQUIRE(result.trace.size() == 3);
    CHECK(result.trace[0].name == "start");
    CHECK(result.trace[1].name == "middle");
    CHECK(result.trace[2].name == "final");
}

TEST_CASE("task graph infers dependencies from produced and consumed keys")
{
    cvkit::infer::detail::TaskGraph graph{};
    graph.add_node(
        std::make_shared<TestNode>(
            "consumer",
            std::vector<std::string>{"scratch:features"},
            std::vector<std::string>{"output:done"}),
        std::vector<std::string>{});
    graph.add_node(
        std::make_shared<TestNode>(
            "producer",
            std::vector<std::string>{"input:image"},
            std::vector<std::string>{"scratch:features"}),
        std::vector<std::string>{});

    const auto metadata = graph.metadata();
    REQUIRE(metadata.size() == 2);
    CHECK(metadata[0].name == "consumer");
    CHECK(metadata[0].consumes == std::vector<std::string>{"scratch:features"});
    CHECK(metadata[0].produces == std::vector<std::string>{"output:done"});
    CHECK(metadata[1].name == "producer");
    CHECK(metadata[1].produces == std::vector<std::string>{"scratch:features"});

    const auto boundary = graph.boundary();
    CHECK(boundary.inputs == std::vector<std::string>{"input:image"});
    CHECK(boundary.outputs == std::vector<std::string>{"output:done"});

    cvkit::infer::detail::Packet packet{};
    packet.input.add("image", cvkit::core::Frame{});
    const auto result = graph.run_sync(std::move(packet));

    REQUIRE(result.trace.size() == 2);
    CHECK(result.trace[0].name == "producer");
    CHECK(result.trace[1].name == "consumer");
}

TEST_CASE("task graph submit_packet records trace for async nodes")
{
    cvkit::infer::detail::TaskGraph graph{};
    graph.add_node(std::make_shared<AsyncTestNode>("async_first"));
    graph.add_node(std::make_shared<TestNode>("second"));

    cvkit::infer::detail::Packet packet{};
    packet.input.add("image", cvkit::core::Frame{});

    auto future = graph.submit_packet(std::move(packet));
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);

    const auto result = future.get();
    REQUIRE(result.trace.size() == 2);
    CHECK(result.trace[0].name == "async_first");
    CHECK(result.trace[0].sequence == 0);
    CHECK(result.trace[0].duration_us >= 0);
    CHECK(result.trace[1].name == "second");
    CHECK(result.trace[1].sequence == 1);
    CHECK(result.trace[1].duration_us >= 0);
}

TEST_CASE("task graph trace records node error messages")
{
    cvkit::infer::detail::TaskGraph graph{};
    graph.add_node(std::make_shared<ErrorTraceNode>("error_node"));
    cvkit::infer::detail::Packet packet{};
    const auto                   result = graph.run_sync(std::move(packet));
    REQUIRE(result.trace.size() == 1);
    CHECK(result.trace[0].name == "error_node");
    CHECK_FALSE(result.trace[0].ok);
    CHECK(result.trace[0].message == "cuda_device preprocess path is not implemented yet");
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

    const auto  sync_output     = model.run_sync(input);
    const auto* sync_detections = sync_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(sync_detections != nullptr);
    REQUIRE_FALSE(sync_detections->empty());

    auto future = model.submit(input);
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);

    const auto  async_output     = future.get();
    const auto* async_detections = async_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(async_detections != nullptr);
    CHECK(async_detections->size() == sync_detections->size());
}

TEST_CASE("model submit uses TensorRT backend asynchronously for yolo detection")
{
#if defined(CVKIT_WITH_TENSORRT)
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to enable TensorRT async smoke validation");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(labels_path));
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path   = model_path.string();
    spec.labels_path  = labels_path.string();
    spec.backend      = cvkit::infer::Backend::tensorrt;
    spec.task         = cvkit::infer::TaskKind::detection;
    spec.family       = "yolo11";
    spec.cache_policy = cvkit::infer::CachePolicy::default_policy;
    spec.tensorrt_profiles.push_back({
        "images",
        {
            {1, 3, 320, 320},
            {1, 3, 640, 640},
            {1, 3, 1280, 1280},
        },
    });

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

    cvkit::infer::TaskInput input{};
    input.add("image", frame);

    auto future = model.submit(input);
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);

    const auto  output     = future.get();
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    REQUIRE_FALSE(detections->empty());
#else
    SKIP("TensorRT backend is not enabled in this build");
#endif
}

TEST_CASE("model run_sync accepts cuda image input for yolo detection")
{
#if defined(CVKIT_WITH_CUDA_RUNTIME)
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
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to validate cuda image input through TensorRT");
    }
    backend = cvkit::infer::Backend::tensorrt;
    #else
    SKIP("no inference backend is enabled in this build");
    #endif

    cvkit::infer::ModelSpec spec{};
    spec.model_path  = model_path.string();
    spec.labels_path = labels_path.string();
    spec.backend     = backend;
    spec.task        = cvkit::infer::TaskKind::detection;
    spec.family      = "yolo11";
    #if defined(CVKIT_WITH_TENSORRT)
    if (backend == cvkit::infer::Backend::tensorrt)
    {
        spec.tensorrt_profiles.push_back({
            .input_name = "images",
            .shape =
                {
                    .min = {1, 3, 320, 320},
                    .opt = {1, 3, 640, 640},
                    .max = {1, 3, 1280, 1280},
                },
        });
    }
    #endif

    cvkit::infer::Model model;
    REQUIRE(model.load(spec));

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    unsigned char* device_image  = nullptr;
    const auto     storage_bytes = static_cast<std::size_t>(image.step) * static_cast<std::size_t>(image.rows);
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_image), storage_bytes) == cudaSuccess);

    struct DeviceImageGuard
    {
        unsigned char* ptr{nullptr};
        ~DeviceImageGuard()
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
    } device_image_guard{device_image};

    REQUIRE(
        cudaMemcpy2D(
            device_image,
            image.step,
            image.data,
            image.step,
            static_cast<std::size_t>(image.cols) * image.elemSize(),
            image.rows,
            cudaMemcpyHostToDevice) == cudaSuccess);

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();

    cvkit::infer::ImageValue cuda_image{};
    cuda_image.frame            = frame;
    cuda_image.memory_device    = cvkit::infer::MemoryDevice::cuda;
    cuda_image.device           = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    cuda_image.storage          = cvkit::infer::StorageKind::external_view;
    cuda_image.row_stride_bytes = image.step;
    cuda_image.external_data    = device_image;
    cuda_image.storage_bytes    = storage_bytes;
    REQUIRE(cuda_image.has_valid_device_view());

    cvkit::infer::TaskInput input{};
    input.add("image", cuda_image);

    cvkit::infer::TaskInput host_input{};
    host_input.add("image", cvkit::infer::ImageValue{frame});
    const auto  host_output     = model.run_sync(host_input);
    const auto* host_detections = host_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(host_detections != nullptr);

    const auto  output     = model.run_sync(input);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    CHECK(detections->size() == host_detections->size());

    const auto trace = model.last_graph_trace();
    REQUIRE_FALSE(trace.empty());
    CHECK(trace.front().ok);
    CHECK(trace.front().message.empty());
#else
    SKIP("CUDA runtime is not enabled in this build");
#endif
}

TEST_CASE("promptable segmentation runs efficient_sam encoder and decoder and returns mask")
{
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto decoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_decoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path) || !std::filesystem::exists(decoder_path))
    {
        SKIP("efficient_sam models are not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path     = encoder_path.string();
    spec.aux_model_path = decoder_path.string();
    spec.backend        = cvkit::infer::Backend::onnxruntime;
    spec.task           = cvkit::infer::TaskKind::promptable_segmentation;
    spec.family         = "efficient_sam";

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        SKIP("onnxruntime promptable segmentation backend is not available in this build");
    }
    REQUIRE(model.loaded());
    REQUIRE(model.task() == cvkit::infer::TaskKind::promptable_segmentation);

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
    input.add("points", std::vector<cvkit::core::Point2f>{{image.cols * 0.5F, image.rows * 0.5F}});
    input.add("point_labels", std::vector<float>{1.0F});

    const auto  output        = model.run_sync(input);
    const auto* mask          = output.find<cvkit::infer::MaskValue>("mask");
    const auto* low_res_masks = output.find<cvkit::infer::TensorValue>("low_res_masks");
    const auto* logits        = output.find<cvkit::infer::TensorValue>("logits");
    const auto* scores        = output.find<std::vector<float>>("scores");
    REQUIRE(mask != nullptr);
    REQUIRE(low_res_masks != nullptr);
    REQUIRE(logits != nullptr);
    REQUIRE(scores != nullptr);
    CHECK(mask->frame.desc.width == frame.desc.width);
    CHECK(mask->frame.desc.height == frame.desc.height);
    CHECK(mask->frame.desc.channels == 1);
    CHECK_FALSE(mask->frame.data.empty());
    CHECK(low_res_masks->shape == std::vector<std::int64_t>{1, 1, 3, 256, 256});
    CHECK(logits->shape == std::vector<std::int64_t>{1, 256, 256});
    CHECK_FALSE(low_res_masks->data.empty());
    CHECK_FALSE(logits->data.empty());
    CHECK(scores->size() == 3);
}

TEST_CASE("promptable segmentation encoder-only returns image embeddings")
{
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path))
    {
        SKIP("efficient_sam encoder model is not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path = encoder_path.string();
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    spec.family     = "efficient_sam_encoder";

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        SKIP("onnxruntime promptable segmentation encoder backend is not available in this build");
    }

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

    const auto  output     = model.run_sync(input);
    const auto* embeddings = output.find<cvkit::infer::TensorValue>("image_embeddings");
    REQUIRE(embeddings != nullptr);
    CHECK(embeddings->shape == std::vector<std::int64_t>{1, 256, 64, 64});
    CHECK_FALSE(embeddings->data.empty());
}

TEST_CASE("promptable segmentation encoder-only accepts cuda image input and returns image embeddings")
{
#if defined(CVKIT_WITH_CUDA_RUNTIME)
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path))
    {
        SKIP("efficient_sam encoder model is not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path = encoder_path.string();
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    spec.family     = "efficient_sam_encoder";
    spec.device     = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        SKIP("onnxruntime promptable segmentation encoder backend is not available in this build");
    }

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    unsigned char* device_image  = nullptr;
    const auto     storage_bytes = static_cast<std::size_t>(image.step) * static_cast<std::size_t>(image.rows);
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_image), storage_bytes) == cudaSuccess);

    struct DeviceImageGuard
    {
        unsigned char* ptr{nullptr};
        ~DeviceImageGuard()
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
    } device_image_guard{device_image};

    REQUIRE(
        cudaMemcpy2D(
            device_image,
            image.step,
            image.data,
            image.step,
            static_cast<std::size_t>(image.cols) * image.elemSize(),
            image.rows,
            cudaMemcpyHostToDevice) == cudaSuccess);

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();

    cvkit::infer::ImageValue host_image{};
    host_image.frame            = frame;
    host_image.row_stride_bytes = image.step;
    host_image.frame.data.assign(image.data, image.data + image.total() * image.elemSize());
    REQUIRE(host_image.has_valid_host_layout());

    cvkit::infer::ImageValue cuda_image{};
    cuda_image.frame            = frame;
    cuda_image.memory_device    = cvkit::infer::MemoryDevice::cuda;
    cuda_image.device           = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    cuda_image.storage          = cvkit::infer::StorageKind::external_view;
    cuda_image.row_stride_bytes = image.step;
    cuda_image.external_data    = device_image;
    cuda_image.storage_bytes    = storage_bytes;
    REQUIRE(cuda_image.has_valid_device_view());

    cvkit::infer::TaskInput host_input{};
    host_input.add("image", host_image);
    const auto  host_output     = model.run_sync(host_input);
    const auto* host_embeddings = host_output.find<cvkit::infer::TensorValue>("image_embeddings");
    REQUIRE(host_embeddings != nullptr);
    REQUIRE_FALSE(host_embeddings->data.empty());

    cvkit::infer::TaskInput cuda_input{};
    cuda_input.add("image", cuda_image);
    const auto  output     = model.run_sync(cuda_input);
    const auto* embeddings = output.find<cvkit::infer::TensorValue>("image_embeddings");
    REQUIRE(embeddings != nullptr);
    CHECK(embeddings->shape == std::vector<std::int64_t>{1, 256, 64, 64});
    CHECK(embeddings->shape == host_embeddings->shape);
    REQUIRE(embeddings->data.size() == host_embeddings->data.size());
    CHECK(embeddings->data.front() == Catch::Approx(host_embeddings->data.front()).margin(2e-3F));
    CHECK(embeddings->data.back() == Catch::Approx(host_embeddings->data.back()).margin(2e-3F));

    const auto trace = model.last_graph_trace();
    REQUIRE_FALSE(trace.empty());
    CHECK(trace.front().ok);
    CHECK(trace.front().message.empty());
#else
    SKIP("CUDA runtime is not enabled in this build");
#endif
}

TEST_CASE("promptable segmentation combined accepts cuda image input and returns mask")
{
#if defined(CVKIT_WITH_CUDA_RUNTIME)
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto decoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_decoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path) || !std::filesystem::exists(decoder_path))
    {
        SKIP("efficient_sam models are not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path     = encoder_path.string();
    spec.aux_model_path = decoder_path.string();
    spec.backend        = cvkit::infer::Backend::onnxruntime;
    spec.task           = cvkit::infer::TaskKind::promptable_segmentation;
    spec.family         = "efficient_sam";
    spec.device         = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        SKIP("onnxruntime promptable segmentation combined backend is not available in this build");
    }

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    unsigned char* device_image  = nullptr;
    const auto     storage_bytes = static_cast<std::size_t>(image.step) * static_cast<std::size_t>(image.rows);
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_image), storage_bytes) == cudaSuccess);

    struct DeviceImageGuard
    {
        unsigned char* ptr{nullptr};
        ~DeviceImageGuard()
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
    } device_image_guard{device_image};

    REQUIRE(
        cudaMemcpy2D(
            device_image,
            image.step,
            image.data,
            image.step,
            static_cast<std::size_t>(image.cols) * image.elemSize(),
            image.rows,
            cudaMemcpyHostToDevice) == cudaSuccess);

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();

    cvkit::infer::ImageValue host_image{};
    host_image.frame            = frame;
    host_image.row_stride_bytes = image.step;
    host_image.frame.data.assign(image.data, image.data + image.total() * image.elemSize());
    REQUIRE(host_image.has_valid_host_layout());

    cvkit::infer::ImageValue cuda_image{};
    cuda_image.frame            = frame;
    cuda_image.memory_device    = cvkit::infer::MemoryDevice::cuda;
    cuda_image.device           = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    cuda_image.storage          = cvkit::infer::StorageKind::external_view;
    cuda_image.row_stride_bytes = image.step;
    cuda_image.external_data    = device_image;
    cuda_image.storage_bytes    = storage_bytes;
    REQUIRE(cuda_image.has_valid_device_view());

    const auto              prompt_points = std::vector<cvkit::core::Point2f>{{image.cols * 0.5F, image.rows * 0.5F}};
    const auto              point_labels  = std::vector<float>{1.0F};

    cvkit::infer::TaskInput host_input{};
    host_input.add("image", host_image);
    host_input.add("points", prompt_points);
    host_input.add("point_labels", point_labels);
    const auto  host_output = model.run_sync(host_input);
    const auto* host_mask   = host_output.find<cvkit::infer::MaskValue>("mask");
    const auto* host_scores = host_output.find<std::vector<float>>("scores");
    REQUIRE(host_mask != nullptr);
    REQUIRE(host_scores != nullptr);
    REQUIRE_FALSE(host_mask->frame.data.empty());

    cvkit::infer::TaskInput cuda_input{};
    cuda_input.add("image", cuda_image);
    cuda_input.add("points", prompt_points);
    cuda_input.add("point_labels", point_labels);
    const auto  output = model.run_sync(cuda_input);
    const auto* mask   = output.find<cvkit::infer::MaskValue>("mask");
    const auto* scores = output.find<std::vector<float>>("scores");
    REQUIRE(mask != nullptr);
    REQUIRE(scores != nullptr);
    CHECK(mask->frame.desc.width == host_mask->frame.desc.width);
    CHECK(mask->frame.desc.height == host_mask->frame.desc.height);
    CHECK(mask->frame.data.size() == host_mask->frame.data.size());
    REQUIRE(scores->size() == host_scores->size());
    CHECK(scores->front() == Catch::Approx(host_scores->front()).margin(2e-3F));
    CHECK(scores->back() == Catch::Approx(host_scores->back()).margin(2e-3F));

    const auto trace = model.last_graph_trace();
    REQUIRE_FALSE(trace.empty());
    CHECK(trace.front().ok);
    CHECK(trace.front().message.empty());
#else
    SKIP("CUDA runtime is not enabled in this build");
#endif
}

TEST_CASE("promptable segmentation decoder-only consumes embeddings and returns mask")
{
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto decoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_decoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path) || !std::filesystem::exists(decoder_path))
    {
        SKIP("efficient_sam models are not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();
    frame.data.assign(image.data, image.data + image.total() * image.elemSize());

    cvkit::infer::ModelSpec encoder_spec{};
    encoder_spec.model_path = encoder_path.string();
    encoder_spec.backend    = cvkit::infer::Backend::onnxruntime;
    encoder_spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    encoder_spec.family     = "efficient_sam_encoder";

    cvkit::infer::Model encoder;
    if (!encoder.load(encoder_spec))
    {
        SKIP("onnxruntime promptable segmentation encoder backend is not available in this build");
    }

    cvkit::infer::TaskInput encoder_input{};
    encoder_input.add("image", frame);
    const auto  encoder_output = encoder.run_sync(encoder_input);
    const auto* embeddings     = encoder_output.find<cvkit::infer::TensorValue>("image_embeddings");
    REQUIRE(embeddings != nullptr);

    cvkit::infer::ModelSpec decoder_spec{};
    decoder_spec.model_path = decoder_path.string();
    decoder_spec.backend    = cvkit::infer::Backend::onnxruntime;
    decoder_spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    decoder_spec.family     = "efficient_sam_decoder";
    decoder_spec.device     = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};

    cvkit::infer::Model decoder;
    if (!decoder.load(decoder_spec))
    {
        SKIP("onnxruntime promptable segmentation decoder backend is not available in this build");
    }

    cvkit::infer::TaskInput decoder_input{};
    decoder_input.add("image", frame);
    decoder_input.add("image_embeddings", *embeddings);
    decoder_input.add("points", std::vector<cvkit::core::Point2f>{{image.cols * 0.5F, image.rows * 0.5F}});
    decoder_input.add("point_labels", std::vector<float>{1.0F});

    const auto  decoder_output = decoder.run_sync(decoder_input);
    const auto* mask           = decoder_output.find<cvkit::infer::MaskValue>("mask");
    const auto* low_res_masks  = decoder_output.find<cvkit::infer::TensorValue>("low_res_masks");
    const auto* logits         = decoder_output.find<cvkit::infer::TensorValue>("logits");
    const auto* scores         = decoder_output.find<std::vector<float>>("scores");
    REQUIRE(mask != nullptr);
    REQUIRE(low_res_masks != nullptr);
    REQUIRE(logits != nullptr);
    REQUIRE(scores != nullptr);
    CHECK(mask->frame.desc.width == frame.desc.width);
    CHECK(mask->frame.desc.height == frame.desc.height);
    CHECK_FALSE(mask->frame.data.empty());
    CHECK(low_res_masks->shape == std::vector<std::int64_t>{1, 1, 3, 256, 256});
    CHECK(logits->shape == std::vector<std::int64_t>{1, 256, 256});
    CHECK(scores->size() == 3);
}

TEST_CASE("promptable segmentation decoder-only consumes cuda embeddings and returns mask")
{
#if defined(CVKIT_WITH_CUDA_RUNTIME)
    const auto source_root  = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto encoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_encoder.sim.onnx";
    const auto decoder_path = source_root / "assets" / "models" / "efficient_sam_vitt_decoder.sim.onnx";
    const auto image_path   = source_root / "assets" / "images" / "test_001.jpg";

    if (!std::filesystem::exists(encoder_path) || !std::filesystem::exists(decoder_path))
    {
        SKIP("efficient_sam models are not present under assets/models");
    }
    REQUIRE(std::filesystem::exists(image_path));

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();
    frame.data.assign(image.data, image.data + image.total() * image.elemSize());

    cvkit::infer::ModelSpec encoder_spec{};
    encoder_spec.model_path = encoder_path.string();
    encoder_spec.backend    = cvkit::infer::Backend::onnxruntime;
    encoder_spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    encoder_spec.family     = "efficient_sam_encoder";

    cvkit::infer::Model encoder;
    if (!encoder.load(encoder_spec))
    {
        SKIP("onnxruntime promptable segmentation encoder backend is not available in this build");
    }

    cvkit::infer::TaskInput encoder_input{};
    encoder_input.add("image", frame);
    const auto  encoder_output = encoder.run_sync(encoder_input);
    const auto* embeddings     = encoder_output.find<cvkit::infer::TensorValue>("image_embeddings");
    REQUIRE(embeddings != nullptr);
    REQUIRE_FALSE(embeddings->data.empty());

    float* device_buffer = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_buffer), embeddings->packed_byte_size()) == cudaSuccess);

    struct DeviceEmbeddingGuard
    {
        float* ptr{nullptr};
        ~DeviceEmbeddingGuard()
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
    } device_embedding_guard{device_buffer};

    REQUIRE(
        cudaMemcpy(
            device_buffer,
            embeddings->data.data(),
            embeddings->packed_byte_size(),
            cudaMemcpyHostToDevice) == cudaSuccess);

    cvkit::infer::TensorValue cuda_embeddings = *embeddings;
    cuda_embeddings.data.clear();
    cuda_embeddings.memory_device = cvkit::infer::MemoryDevice::cuda;
    cuda_embeddings.storage       = cvkit::infer::StorageKind::external_view;
    cuda_embeddings.external_data = device_buffer;
    cuda_embeddings.storage_bytes = embeddings->packed_byte_size();
    REQUIRE(cuda_embeddings.has_valid_device_view());

    cvkit::infer::ModelSpec decoder_spec{};
    decoder_spec.model_path = decoder_path.string();
    decoder_spec.backend    = cvkit::infer::Backend::onnxruntime;
    decoder_spec.task       = cvkit::infer::TaskKind::promptable_segmentation;
    decoder_spec.family     = "efficient_sam_decoder";

    cvkit::infer::Model decoder;
    if (!decoder.load(decoder_spec))
    {
        SKIP("onnxruntime promptable segmentation decoder backend is not available in this build");
    }

    cvkit::infer::TaskInput decoder_input{};
    decoder_input.add("image", frame);
    decoder_input.add("image_embeddings", cuda_embeddings);
    decoder_input.add("points", std::vector<cvkit::core::Point2f>{{image.cols * 0.5F, image.rows * 0.5F}});
    decoder_input.add("point_labels", std::vector<float>{1.0F});

    const auto  decoder_output = decoder.run_sync(decoder_input);
    const auto* mask           = decoder_output.find<cvkit::infer::MaskValue>("mask");
    const auto* scores         = decoder_output.find<std::vector<float>>("scores");
    REQUIRE(mask != nullptr);
    REQUIRE(scores != nullptr);
    CHECK(mask->frame.desc.width == frame.desc.width);
    CHECK(mask->frame.desc.height == frame.desc.height);
    CHECK_FALSE(mask->frame.data.empty());
    CHECK(scores->size() == 3);
#else
    SKIP("CUDA runtime is not enabled in this build");
#endif
}

#if defined(CVKIT_WITH_TENSORRT)
TEST_CASE("tensorrt model supports preferred cuda output metadata without breaking detection")
{
    #if defined(CVKIT_WITH_TENSORRT) && defined(CVKIT_WITH_CUDA_RUNTIME)
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to validate TensorRT device-output preference");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(labels_path));
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path                     = model_path.string();
    spec.labels_path                    = labels_path.string();
    spec.backend                        = cvkit::infer::Backend::tensorrt;
    spec.task                           = cvkit::infer::TaskKind::detection;
    spec.family                         = "yolo11";
    spec.tensorrt_prefer_device_outputs = true;
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

    const auto session = model.session_info();
    REQUIRE_FALSE(session.outputs.empty());
    CHECK(session.outputs.front().memory_device == cvkit::infer::MemoryDevice::cuda);

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
    #else
    SKIP("TensorRT backend or CUDA runtime is not enabled in this build");
    #endif
}

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

TEST_CASE("tensorrt face detection supports scrfd raw-bgr tiled public api")
{
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to enable the TensorRT SCRFD tiled smoke test");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "scrfd_10g_ac133ba7.onnx";
    const auto image_path  = source_root / "assets" / "images" / "face.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(image_path));

    cvkit::infer::ModelSpec spec{};
    spec.model_path = model_path.string();
    spec.backend    = cvkit::infer::Backend::tensorrt;
    spec.task       = cvkit::infer::TaskKind::face_detection;
    spec.family     = "scrfd_raw_bgr";

    cvkit::infer::Model model;
    model.set_confidence_threshold(0.5F);
    model.set_iou_threshold(0.4F);
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

    cvkit::infer::TaskInput input{};
    input.add("image", frame);

    cvkit::infer::TileOptions tile_options{};
    tile_options.enabled     = true;
    tile_options.tile_width  = 640;
    tile_options.tile_height = 640;
    tile_options.overlap_x   = 160;
    tile_options.overlap_y   = 160;
    model.set_tile_options(tile_options);

    const auto  output     = model.run_sync(input);
    const auto* detections = output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(detections != nullptr);
    CHECK(detections->size() > 500U);
    CHECK(std::all_of(
        detections->begin(),
        detections->end(),
        [&frame](const cvkit::core::Detection& detection)
        {
            return detection.box.x >= 0.0F &&
                   detection.box.y >= 0.0F &&
                   detection.box.x <= static_cast<float>(frame.desc.width) &&
                   detection.box.y <= static_cast<float>(frame.desc.height);
        }));
    CHECK(std::any_of(
        detections->begin(),
        detections->end(),
        [](const cvkit::core::Detection& detection)
        { return detection.keypoints.size() == 5; }));
}

TEST_CASE("tensorrt detection public api covers sync async cuda input and preferred device outputs")
{
    #if defined(CVKIT_WITH_CUDA_RUNTIME)
    if (std::getenv("CVKIT_RUN_TENSORRT_SMOKE") == nullptr)
    {
        SKIP("set CVKIT_RUN_TENSORRT_SMOKE=1 to run the full TensorRT public-api smoke");
    }

    const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto model_path  = source_root / "assets" / "models" / "yolo11n.onnx";
    const auto labels_path = source_root / "assets" / "labels" / "coco80.txt";
    const auto image_path  = source_root / "assets" / "images" / "test_001.jpg";

    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(labels_path));
    REQUIRE(std::filesystem::exists(image_path));

    const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(image.empty());

    cvkit::core::Frame frame{};
    frame.desc.width    = image.cols;
    frame.desc.height   = image.rows;
    frame.desc.channels = image.channels();
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.source        = image_path.string();
    frame.data.assign(image.data, image.data + image.total() * image.elemSize());

    cvkit::infer::TaskInput host_input{};
    host_input.add("image", cvkit::infer::ImageValue{frame});

    cvkit::infer::ModelSpec base_spec{};
    base_spec.model_path  = model_path.string();
    base_spec.labels_path = labels_path.string();
    base_spec.backend     = cvkit::infer::Backend::tensorrt;
    base_spec.task        = cvkit::infer::TaskKind::detection;
    base_spec.family      = "yolo11";
    base_spec.tensorrt_profiles.push_back({
        .input_name = "images",
        .shape =
            {
                .min = {1, 3, 320, 320},
                .opt = {1, 3, 640, 640},
                .max = {1, 3, 1280, 1280},
            },
    });

    cvkit::infer::Model model;
    REQUIRE(model.load(base_spec));
    REQUIRE(model.loaded());
    REQUIRE(model.backend() == cvkit::infer::Backend::tensorrt);

    const auto  sync_output     = model.run_sync(host_input);
    const auto* sync_detections = sync_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(sync_detections != nullptr);
    REQUIRE_FALSE(sync_detections->empty());

    auto future = model.submit(host_input);
    REQUIRE(future.valid());
    CHECK(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);

    const auto  async_output     = future.get();
    const auto* async_detections = async_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(async_detections != nullptr);
    CHECK(async_detections->size() == sync_detections->size());

    unsigned char* device_image  = nullptr;
    const auto     storage_bytes = static_cast<std::size_t>(image.step) * static_cast<std::size_t>(image.rows);
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_image), storage_bytes) == cudaSuccess);

    struct DeviceImageGuard
    {
        unsigned char* ptr{nullptr};
        ~DeviceImageGuard()
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
    } device_image_guard{device_image};

    REQUIRE(
        cudaMemcpy2D(
            device_image,
            image.step,
            image.data,
            image.step,
            static_cast<std::size_t>(image.cols) * image.elemSize(),
            image.rows,
            cudaMemcpyHostToDevice) == cudaSuccess);

    cvkit::infer::ImageValue cuda_image{};
    cuda_image.frame            = frame;
    cuda_image.memory_device    = cvkit::infer::MemoryDevice::cuda;
    cuda_image.device           = cvkit::infer::DeviceRef{cvkit::infer::DeviceKind::cuda, 0};
    cuda_image.storage          = cvkit::infer::StorageKind::external_view;
    cuda_image.row_stride_bytes = image.step;
    cuda_image.external_data    = device_image;
    cuda_image.storage_bytes    = storage_bytes;
    REQUIRE(cuda_image.has_valid_device_view());

    cvkit::infer::TaskInput cuda_input{};
    cuda_input.add("image", cuda_image);

    cvkit::infer::Model cuda_model;
    REQUIRE(cuda_model.load(base_spec));
    const auto  cuda_output     = cuda_model.run_sync(cuda_input);
    const auto* cuda_detections = cuda_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(cuda_detections != nullptr);
    const auto cuda_trace = cuda_model.last_graph_trace();
    REQUIRE_FALSE(cuda_trace.empty());
    CHECK(cuda_trace.front().ok);
    CHECK(cuda_trace.front().message.empty());

    auto preferred_spec                           = base_spec;
    preferred_spec.tensorrt_prefer_device_outputs = true;

    cvkit::infer::Model preferred_model;
    REQUIRE(preferred_model.load(preferred_spec));
    const auto session_info = preferred_model.session_info();
    REQUIRE_FALSE(session_info.outputs.empty());
    CHECK(session_info.outputs.front().memory_device == cvkit::infer::MemoryDevice::cuda);

    const auto  preferred_output = preferred_model.run_sync(host_input);
    const auto* preferred_detections =
        preferred_output.find<std::vector<cvkit::core::Detection>>("detections");
    REQUIRE(preferred_detections != nullptr);
    CHECK(preferred_detections->size() == sync_detections->size());
    #else
    SKIP("CUDA runtime is not enabled in this build");
    #endif
}
#endif
