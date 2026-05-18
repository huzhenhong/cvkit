#include "classification_pipeline.h"

#include "cvkit/infer/tasks/classification.h"

#include "../../utils/image_value_utils.h"
#include "../../utils/opencv_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] std::optional<cvkit::core::Frame> resolve_input_frame(
            const TaskInput& input)
        {
            if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
            {
                return materialize_host_frame(*image);
            }
            if (const auto* frame = input.find<cvkit::core::Frame>("image"); frame != nullptr)
            {
                return *frame;
            }
            return std::nullopt;
        }

        [[nodiscard]] std::vector<std::int64_t> resolve_input_shape(
            const IBackendSession&    backend,
            const cvkit::core::Frame& frame)
        {
            if (const auto* info = backend.input_info(0); info != nullptr && info->shape.size() == 4)
            {
                auto shape = info->shape;
                if (shape[2] <= 0)
                {
                    shape[2] = frame.desc.height;
                }
                if (shape[3] <= 0)
                {
                    shape[3] = frame.desc.width;
                }
                return shape;
            }
            return {1, 3, frame.desc.height, frame.desc.width};
        }

        [[nodiscard]] RawTensor build_classification_input(
            const cvkit::core::Frame&        frame,
            const std::vector<std::int64_t>& input_shape)
        {
            RawTensor tensor{};
            tensor.name  = "images";
            tensor.shape = input_shape;

            if (input_shape.size() != 4 || input_shape[1] != 3 || input_shape[2] <= 0 || input_shape[3] <= 0)
            {
                return tensor;
            }

            auto source = frame_to_mat_copy(frame);
            if (source.empty())
            {
                return tensor;
            }

            cv::Mat rgb;
            if (source.channels() == 3)
            {
                if (frame.desc.format == cvkit::core::PixelFormat::rgb8)
                {
                    rgb = source;
                }
                else
                {
                    cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
                }
            }
            else
            {
                cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
            }

            cv::Mat resized;
            cv::resize(
                rgb,
                resized,
                cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])),
                0.0,
                0.0,
                cv::INTER_LINEAR);
            resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

            const auto width      = static_cast<int>(input_shape[3]);
            const auto height     = static_cast<int>(input_shape[2]);
            const auto plane_size = static_cast<std::size_t>(width * height);
            tensor.data.resize(plane_size * 3U);

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const auto pixel                       = resized.at<cv::Vec3f>(y, x);
                    const auto offset                      = static_cast<std::size_t>(y * width + x);
                    tensor.data[offset]                    = pixel[0];
                    tensor.data[plane_size + offset]       = pixel[1];
                    tensor.data[(2 * plane_size) + offset] = pixel[2];
                }
            }

            return tensor;
        }

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
        const auto frame = resolve_input_frame(input);
        if (!frame.has_value())
        {
            return {};
        }

        const auto input_shape  = resolve_input_shape(backend, *frame);
        auto       input_tensor = build_classification_input(*frame, input_shape);
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
