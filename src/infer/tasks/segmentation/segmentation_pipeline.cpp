#include "segmentation_pipeline.h"

#include "cvkit/infer/tasks/segmentation.h"

#include "../../utils/image_preprocess.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] cvkit::core::Frame build_argmax_mask(
            const RawTensor&          logits,
            const cvkit::core::Frame& source_frame)
        {
            cvkit::core::Frame mask{};
            mask.desc.width    = source_frame.desc.width;
            mask.desc.height   = source_frame.desc.height;
            mask.desc.channels = 1;
            mask.desc.format   = cvkit::core::PixelFormat::unknown;
            mask.source        = source_frame.source;

            if (source_frame.desc.width <= 0 || source_frame.desc.height <= 0 ||
                logits.shape.size() != 4 || logits.data.empty())
            {
                return mask;
            }

            const auto channels = static_cast<int>(logits.shape[1]);
            const auto height   = static_cast<int>(logits.shape[2]);
            const auto width    = static_cast<int>(logits.shape[3]);
            if (channels <= 0 || height <= 0 || width <= 0)
            {
                return mask;
            }

            const auto plane_size = static_cast<std::size_t>(width * height);
            if (logits.data.size() < static_cast<std::size_t>(channels) * plane_size)
            {
                return mask;
            }

            cv::Mat small_mask(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const auto offset     = static_cast<std::size_t>(y * width + x);
                    int        best_class = 0;
                    float      best_score = logits.data[offset];
                    for (int channel = 1; channel < channels; ++channel)
                    {
                        const auto score = logits.data[(static_cast<std::size_t>(channel) * plane_size) + offset];
                        if (score > best_score)
                        {
                            best_score = score;
                            best_class = channel;
                        }
                    }
                    small_mask.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(
                        std::clamp(best_class, 0, 255));
                }
            }

            cv::Mat resized;
            cv::resize(
                small_mask,
                resized,
                cv::Size(source_frame.desc.width, source_frame.desc.height),
                0.0,
                0.0,
                cv::INTER_NEAREST);

            mask.data.assign(resized.data, resized.data + resized.total());
            return mask;
        }

    }  // namespace

    TaskKind SegmentationPipeline::task() const
    {
        return TaskKind::segmentation;
    }

    TaskSchema SegmentationPipeline::schema() const
    {
        return segmentation_schema();
    }

    TaskOutput SegmentationPipeline::run_sync(
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

        auto mask = build_argmax_mask(outputs.front(), *frame);
        if (mask.data.empty())
        {
            return {};
        }

        TaskOutput output{};
        output.add("mask", cvkit::infer::MaskValue{std::move(mask)});
        output.add("logits", outputs.front());
        return output;
    }

}  // namespace cvkit::infer::detail
