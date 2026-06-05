#include "classification_pipeline.h"

#include "cvkit/infer/tasks/classification.h"

#include "../../utils/image_preprocess.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] std::optional<ClassificationValue> build_top1_classification(
            const RawTensorMap&             outputs,
            const std::vector<std::string>& labels,
            FloatListValue*                 scores)
        {
            if (outputs.empty() || outputs.front().data.empty())
            {
                return std::nullopt;
            }

            const auto& logits = outputs.front().data;
            const auto  best   = std::max_element(logits.begin(), logits.end());
            if (best == logits.end())
            {
                return std::nullopt;
            }

            const auto class_id = static_cast<int>(std::distance(logits.begin(), best));
            if (scores != nullptr)
            {
                *scores = logits;
            }

            ClassificationValue result{};
            result.class_id = class_id;
            result.score    = *best;
            if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels.size())
            {
                result.label = labels[static_cast<std::size_t>(class_id)];
            }
            return result;
        }

    }  // namespace

    TaskKind ClassificationPipeline::task() const
    {
        return TaskKind::classification;
    }

    TaskSchema ClassificationPipeline::schema() const
    {
        return classification_schema();
    }

    TaskOutput ClassificationPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        const auto frame = resolve_host_image_input(input);
        if (!frame.has_value())
        {
            return {};
        }

        const auto input_shape  = resolve_nchw_input_shape(backend, *frame);
        auto       input_tensor = build_rgb_nchw_float_input(*frame, input_shape);
        if (input_tensor.data.empty())
        {
            return {};
        }

        RawTensorMap inputs{};
        inputs.push_back(std::move(input_tensor));
        const auto     outputs = backend.run(inputs);

        FloatListValue scores{};
        const auto     classification = build_top1_classification(outputs, context.labels, &scores);
        if (!classification.has_value())
        {
            return {};
        }

        TaskOutput output{};
        output.add("classification", *classification);
        output.add("scores", std::move(scores));
        return output;
    }

}  // namespace cvkit::infer::detail
