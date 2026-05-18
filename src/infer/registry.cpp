#include "cvkit/infer/tasks/classification.h"
#include "cvkit/infer/tasks/detection.h"
#include "cvkit/infer/tasks/promptable_segmentation.h"

#include "backends/backend_session.h"
#include "backends/onnxruntime/ort_session.h"
#include "backends/tensorrt/trt_session.h"
#include "tasks/classification/classification_pipeline.h"
#include "tasks/detection/detection_pipeline.h"
#include "tasks/promptable_segmentation/promptable_segmentation_pipeline.h"

#include <memory>
#include <string_view>

namespace cvkit::infer
{

    TaskSchema detection_schema()
    {
        return TaskSchema{
            TaskKind::detection,
            {IOField{"image", ValueKind::image, false}},
            {IOField{"detections", ValueKind::detections, false}}};
    }

    TaskSchema classification_schema()
    {
        return TaskSchema{
            TaskKind::classification,
            {IOField{"image", ValueKind::image, false}},
            {
                IOField{"classification", ValueKind::classification, false},
                IOField{"scores", ValueKind::floats, true},
            }};
    }

    TaskSchema promptable_segmentation_schema()
    {
        return TaskSchema{
            TaskKind::promptable_segmentation,
            {
                IOField{"image", ValueKind::image, false},
                IOField{"image_embeddings", ValueKind::tensor, true},
                IOField{"points", ValueKind::points2f, true},
                IOField{"point_labels", ValueKind::floats, true},
                IOField{"box", ValueKind::bbox, true},
                IOField{"prompt", ValueKind::text, true},
            },
            {
                IOField{"image_embeddings", ValueKind::tensor, true},
                IOField{"mask", ValueKind::mask, true},
                IOField{"low_res_masks", ValueKind::tensor, true},
                IOField{"logits", ValueKind::tensor, true},
                IOField{"scores", ValueKind::floats, true},
            }};
    }

}  // namespace cvkit::infer

namespace cvkit::infer::detail
{

    std::unique_ptr<IBackendSession> create_backend_session(Backend backend)
    {
        switch (backend)
        {
            case Backend::onnxruntime:
                return std::make_unique<OrtSession>();
            case Backend::tensorrt:
                return std::make_unique<TrtSession>();
            case Backend::none:
            default:
                return nullptr;
        }
    }

    std::unique_ptr<ITaskPipeline> create_task_pipeline(TaskKind task, std::string_view family)
    {
        static_cast<void>(family);
        switch (task)
        {
            case TaskKind::detection:
                return std::make_unique<DetectionPipeline>();
            case TaskKind::classification:
                return std::make_unique<ClassificationPipeline>();
            case TaskKind::promptable_segmentation:
                return std::make_unique<PromptableSegmentationPipeline>();
            case TaskKind::unknown:
            default:
                return nullptr;
        }
    }

}  // namespace cvkit::infer::detail
