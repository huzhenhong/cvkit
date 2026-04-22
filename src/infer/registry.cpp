#include "cvkit/infer/tasks/detection.h"
#include "cvkit/infer/tasks/promptable_segmentation.h"

#include "backends/backend_session.h"
#include "backends/onnxruntime/ort_session.h"
#include "backends/tensorrt/trt_session.h"
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
            {IOField{"image", ValueKind::frame, false}},
            {IOField{"detections", ValueKind::detections, false}}};
    }

    TaskSchema promptable_segmentation_schema()
    {
        return TaskSchema{
            TaskKind::promptable_segmentation,
            {
                IOField{"image", ValueKind::frame, false},
                IOField{"points", ValueKind::points2f, true},
                IOField{"prompt", ValueKind::text, true},
            },
            {
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
            case TaskKind::promptable_segmentation:
                return std::make_unique<PromptableSegmentationPipeline>();
            case TaskKind::unknown:
            default:
                return nullptr;
        }
    }

}  // namespace cvkit::infer::detail
