#include "cvkit/core/types.h"
#include "cvkit/image/ops.h"
#include "cvkit/infer/debug.h"
#include "cvkit/infer/model.h"
#include "cvkit/infer/tasks/classification.h"
#include "cvkit/infer/tasks/detection.h"
#include "cvkit/infer/tasks/face_detection.h"
#include "cvkit/infer/tasks/facemesh.h"
#include "cvkit/infer/tasks/pose.h"
#include "cvkit/infer/tasks/promptable_segmentation.h"
#include "cvkit/infer/tasks/segmentation.h"
#include "cvkit/media/source.h"
#include "cvkit/media/writer.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    bool require(bool condition)
    {
        return condition;
    }

    bool schema_has_io(const cvkit::infer::TaskSchema& schema)
    {
        return !schema.inputs.empty() && !schema.outputs.empty();
    }

}  // namespace

int main()
{
    cvkit::core::Frame frame{};
    frame.desc.width    = 4;
    frame.desc.height   = 4;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(4 * 4 * 3), 7U);

    cvkit::core::DeviceFrame device_frame{};
    device_frame.desc.width    = 4;
    device_frame.desc.height   = 4;
    device_frame.desc.channels = 1;
    device_frame.desc.format   = cvkit::core::PixelFormat::nv12;
    device_frame.memory_device = cvkit::core::MemoryDevice::cuda;
    device_frame.device_index  = 0;
    if (!require(!device_frame.valid()))
    {
        return 1;
    }

    cvkit::infer::ImageValue device_image = cvkit::infer::image_value_from_device_frame(device_frame);
    if (!require(device_image.memory_device == cvkit::infer::MemoryDevice::cuda &&
                 device_image.device.kind == cvkit::infer::DeviceKind::cuda &&
                 device_image.device.index == 0 &&
                 device_image.frame.desc.format == cvkit::core::PixelFormat::nv12 &&
                 !device_image.has_valid_device_view()))
    {
        return 1;
    }

    const auto resized = cvkit::image::resize(frame, 2, 2);
    if (!require(resized.desc.width == 2 && resized.desc.height == 2))
    {
        return 1;
    }

    cvkit::infer::ModelSpec spec{};
    spec.model_path = "demo.onnx";
    spec.backend    = cvkit::infer::Backend::onnxruntime;
    spec.task       = cvkit::infer::TaskKind::face_detection;
    spec.family     = "scrfd_raw_bgr";

    cvkit::infer::Model model;
    if (!require(!model.loaded()))
    {
        return 1;
    }

    cvkit::infer::TileOptions tile_options{};
    tile_options.enabled     = true;
    tile_options.tile_width  = 640;
    tile_options.tile_height = 640;
    tile_options.overlap_x   = 160;
    tile_options.overlap_y   = 160;
    model.set_tile_options(tile_options);

    const auto stored_tile_options = model.tile_options();
    if (!require(stored_tile_options.enabled &&
                 stored_tile_options.tile_width == 640 &&
                 stored_tile_options.tile_height == 640 &&
                 stored_tile_options.overlap_x == 160 &&
                 stored_tile_options.overlap_y == 160))
    {
        return 1;
    }

    cvkit::infer::GraphTraceInfo trace_info{};
    trace_info.name                 = "tile_0";
    trace_info.has_tile_info        = true;
    trace_info.tile_info.tile_index = 0;
    trace_info.tile_info.width      = 640;
    trace_info.tile_info.height     = 640;
    if (!require(trace_info.has_tile_info &&
                 trace_info.tile_info.tile_index == 0 &&
                 trace_info.tile_info.width == 640 &&
                 trace_info.tile_info.height == 640))
    {
        return 1;
    }

    cvkit::infer::TaskInput input{};
    input.add("image", cvkit::infer::ImageValue{frame});
    input.add("tag", std::string{"package-smoke"});
    if (!require(input.find<cvkit::infer::ImageValue>("image") != nullptr &&
                 input.find<std::string>("tag") != nullptr))
    {
        return 1;
    }

    cvkit::core::Detection detection{};
    detection.score = 0.9F;
    detection.keypoints.push_back(cvkit::core::Point2f{1.0F, 2.0F});

    cvkit::infer::TaskOutput output{};
    output.add("detections", std::vector<cvkit::core::Detection>{detection});
    if (!require(output.find<std::vector<cvkit::core::Detection>>("detections") != nullptr))
    {
        return 1;
    }

    const std::vector<cvkit::infer::TaskSchema> schemas{
        cvkit::infer::detection_schema(),
        cvkit::infer::face_detection_schema(),
        cvkit::infer::classification_schema(),
        cvkit::infer::segmentation_schema(),
        cvkit::infer::pose_schema(),
        cvkit::infer::facemesh_schema(),
        cvkit::infer::promptable_segmentation_schema(),
    };
    for (const auto& schema : schemas)
    {
        if (!require(schema.task != cvkit::infer::TaskKind::unknown && schema_has_io(schema)))
        {
            return 1;
        }
    }

    if (!require(cvkit::infer::task_name(cvkit::infer::TaskKind::face_detection) == std::string_view{"face_detection"} &&
                 cvkit::infer::backend_name(cvkit::infer::Backend::onnxruntime) == std::string_view{"onnxruntime"}))
    {
        return 1;
    }

    cvkit::media::Source source;
    if (!require(!source.open(std::string{})))
    {
        return 1;
    }
    if (!require(!source.is_open() &&
                 source.status() == cvkit::media::SourceStatus::invalid_uri &&
                 !source.status_message().empty() &&
                 !source.info().open))
    {
        return 1;
    }

    const auto media_capabilities = cvkit::media::runtime_capabilities(7);
    if (!require(!media_capabilities.gstreamer || media_capabilities.gstreamer_appsink))
    {
        return 1;
    }

    cvkit::media::WriterOptions writer_options{};
    writer_options.uri     = {};
    writer_options.backend = cvkit::media::WriterBackend::opencv;
    writer_options.width   = 4;
    writer_options.height  = 4;
    writer_options.fps     = 25.0;

    cvkit::media::Writer writer;
    if (!require(!writer.open(writer_options) &&
                 writer.status() == cvkit::media::WriterStatus::invalid_options &&
                 !writer.status_message().empty() &&
                 !writer.info().open))
    {
        return 1;
    }

    return 0;
}
