#include "pose_pipeline.h"

#include "cvkit/infer/tasks/pose.h"

#include "../../utils/image_preprocess.h"

#include <cstddef>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] std::size_t resolve_keypoint_count(const RawTensor& tensor, int* stride)
        {
            if (stride != nullptr)
            {
                *stride = 0;
            }
            if (tensor.data.empty())
            {
                return 0U;
            }

            if (!tensor.shape.empty())
            {
                const auto last_dim = tensor.shape.back();
                if (last_dim == 2 || last_dim == 3)
                {
                    if (stride != nullptr)
                    {
                        *stride = static_cast<int>(last_dim);
                    }
                    return tensor.data.size() / static_cast<std::size_t>(last_dim);
                }
            }

            if (tensor.data.size() % 3U == 0U)
            {
                if (stride != nullptr)
                {
                    *stride = 3;
                }
                return tensor.data.size() / 3U;
            }
            if (tensor.data.size() % 2U == 0U)
            {
                if (stride != nullptr)
                {
                    *stride = 2;
                }
                return tensor.data.size() / 2U;
            }
            return 0U;
        }

        [[nodiscard]] KeypointsValue build_keypoints(const RawTensor& tensor, FloatListValue* scores)
        {
            KeypointsValue keypoints{};
            int            stride = 0;
            const auto     count  = resolve_keypoint_count(tensor, &stride);
            if (count == 0U || stride < 2)
            {
                return keypoints;
            }

            keypoints.points.reserve(count);
            if (scores != nullptr && stride >= 3)
            {
                scores->reserve(count);
            }

            for (std::size_t index = 0; index < count; ++index)
            {
                const auto offset = index * static_cast<std::size_t>(stride);
                keypoints.points.push_back(cvkit::core::Point2f{
                    tensor.data[offset],
                    tensor.data[offset + 1U]});
                if (scores != nullptr && stride >= 3)
                {
                    scores->push_back(tensor.data[offset + 2U]);
                }
            }
            return keypoints;
        }

    }  // namespace

    TaskKind PosePipeline::task() const
    {
        return TaskKind::pose;
    }

    TaskSchema PosePipeline::schema() const
    {
        return pose_schema();
    }

    TaskOutput PosePipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        static_cast<void>(context);

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
        const auto outputs = backend.run(inputs);
        if (outputs.empty())
        {
            return {};
        }

        FloatListValue scores{};
        auto           keypoints = build_keypoints(outputs.front(), &scores);
        if (keypoints.points.empty())
        {
            return {};
        }

        TensorValue raw{};
        raw.name        = outputs.front().name;
        raw.shape       = outputs.front().shape;
        raw.data        = outputs.front().data;
        raw.data_type   = outputs.front().data_type;
        raw.memory_device = outputs.front().memory_device;

        TaskOutput output{};
        output.add("keypoints", std::move(keypoints));
        if (!scores.empty())
        {
            output.add("scores", std::move(scores));
        }
        output.add("raw", std::move(raw));
        return output;
    }

}  // namespace cvkit::infer::detail
