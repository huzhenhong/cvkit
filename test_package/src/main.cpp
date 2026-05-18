#include "cvkit/image/ops.h"
#include "cvkit/infer/model.h"
#include "cvkit/media/source.h"

int main()
{
    cvkit::media::Source source;
    if (!source.open("file://demo.mp4"))
    {
        return 1;
    }

    cvkit::core::Frame frame;
    if (!source.read(frame))
    {
        return 1;
    }

    auto resized = cvkit::image::resize(frame, 320, 320);

    cvkit::infer::ModelSpec spec{};
    spec.model_path = "demo.onnx";
    spec.backend = cvkit::infer::Backend::none;
    spec.task = cvkit::infer::TaskKind::detection;
    spec.family = "yolo11";

    cvkit::infer::Model model;
    if (!model.load(spec))
    {
        return 1;
    }

    const auto detections = model.run_detection(resized);
    return detections.empty() ? 1 : 0;
}
