#include "detection_pipeline.h"

#include "cvkit/infer/tasks/detection.h"

#include "yolo/yolo_postprocess.h"
#include "yolo/yolo_preprocess_cpu.h"

#include "../../utils/tensor_layout.h"

namespace cvkit::infer::detail
{

    TaskKind DetectionPipeline::task() const
    {
        return TaskKind::detection;
    }

    TaskSchema DetectionPipeline::schema() const
    {
        return detection_schema();
    }

    TaskOutput DetectionPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        TaskOutput output{};
        const auto* frame = input.find<cvkit::core::Frame>("image");
        if (frame == nullptr)
        {
            return output;
        }

        std::vector<std::int64_t> backend_shape{};
        if (const auto* input_info = backend.input_info(0); input_info != nullptr)
        {
            backend_shape = input_info->shape;
        }

        if (backend_shape.empty())
        {
            backend_shape = {1, 3, frame->desc.height, frame->desc.width};
        }

        const auto resolved_shape = resolve_input_shape(backend_shape, *frame);
        auto       preprocess     = preprocess_yolo_cpu(*frame, resolved_shape);
        if (preprocess.tensor.data.empty())
        {
            return output;
        }

        RawTensorMap inputs;
        inputs.push_back(std::move(preprocess.tensor));
        auto detections = postprocess_yolo_detections(
            backend.run(inputs),
            preprocess,
            *frame,
            context.labels,
            context.confidence_threshold,
            context.iou_threshold);

        output.add("detections", std::move(detections));
        return output;
    }

}  // namespace cvkit::infer::detail
