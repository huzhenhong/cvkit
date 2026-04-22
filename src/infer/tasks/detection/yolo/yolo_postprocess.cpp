#include "yolo_postprocess.h"

#include "../../../utils/nms.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace cvkit::infer::detail
{
    namespace
    {

        [[nodiscard]] cvkit::core::BBox decode_yolo_box(
            float                     cx,
            float                     cy,
            float                     width,
            float                     height,
            const LetterboxResult&    preprocess,
            const cvkit::core::Frame& frame)
        {
            if (std::max({cx, cy, width, height}) <= 2.0F)
            {
                cx *= static_cast<float>(preprocess.input_width);
                cy *= static_cast<float>(preprocess.input_height);
                width *= static_cast<float>(preprocess.input_width);
                height *= static_cast<float>(preprocess.input_height);
            }

            const auto        x0 = (cx - width * 0.5F - preprocess.pad_x) / preprocess.scale;
            const auto        y0 = (cy - height * 0.5F - preprocess.pad_y) / preprocess.scale;
            const auto        x1 = (cx + width * 0.5F - preprocess.pad_x) / preprocess.scale;
            const auto        y1 = (cy + height * 0.5F - preprocess.pad_y) / preprocess.scale;

            cvkit::core::BBox box{};
            box.x             = std::clamp(x0, 0.0F, static_cast<float>(frame.desc.width));
            box.y             = std::clamp(y0, 0.0F, static_cast<float>(frame.desc.height));
            const auto right  = std::clamp(x1, 0.0F, static_cast<float>(frame.desc.width));
            const auto bottom = std::clamp(y1, 0.0F, static_cast<float>(frame.desc.height));
            box.width         = std::max(0.0F, right - box.x);
            box.height        = std::max(0.0F, bottom - box.y);
            return box;
        }

        [[nodiscard]] std::vector<Candidate> parse_yolo_output_tensor(
            const RawTensor&          output,
            const LetterboxResult&    preprocess,
            const cvkit::core::Frame& frame,
            float                     confidence_threshold)
        {
            std::vector<Candidate> candidates;
            if (output.data.empty())
            {
                return candidates;
            }

            const auto&                                    shape = output.shape;
            const auto                                     count = output.data.size();
            const auto*                                    data  = output.data.data();

            std::size_t                                    boxes  = 0;
            std::size_t                                    attrs  = 0;
            std::function<float(std::size_t, std::size_t)> access = [&](std::size_t box_index, std::size_t attr_index) -> float
            {
                return data[box_index * attrs + attr_index];
            };
            bool transposed = false;

            if (shape.size() == 3)
            {
                const auto dim1 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[1]));
                const auto dim2 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[2]));
                if (dim1 >= 5 && dim2 >= 5)
                {
                    if (dim1 <= dim2)
                    {
                        attrs      = dim1;
                        boxes      = dim2;
                        transposed = true;
                    }
                    else
                    {
                        boxes = dim1;
                        attrs = dim2;
                    }
                }
            }
            else if (shape.size() == 2)
            {
                const auto dim0 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[0]));
                const auto dim1 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[1]));
                if (dim0 >= 5 && dim1 >= 5)
                {
                    if (dim0 <= dim1)
                    {
                        attrs      = dim0;
                        boxes      = dim1;
                        transposed = true;
                    }
                    else
                    {
                        boxes = dim0;
                        attrs = dim1;
                    }
                }
            }

            if (boxes == 0 || attrs <= 4 || count == 0)
            {
                return candidates;
            }

            if (transposed)
            {
                access = [&](std::size_t box_index, std::size_t attr_index) -> float
                {
                    return data[attr_index * boxes + box_index];
                };
            }

            candidates.reserve(boxes);
            for (std::size_t box_index = 0; box_index < boxes; ++box_index)
            {
                const auto cx     = access(box_index, 0);
                const auto cy     = access(box_index, 1);
                const auto width  = access(box_index, 2);
                const auto height = access(box_index, 3);

                float      best_score = 0.0F;
                int        best_class = -1;
                for (std::size_t class_index = 4; class_index < attrs; ++class_index)
                {
                    const auto score = access(box_index, class_index);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = static_cast<int>(class_index - 4);
                    }
                }

                if (best_class < 0 || best_score < confidence_threshold)
                {
                    continue;
                }

                auto box = decode_yolo_box(cx, cy, width, height, preprocess, frame);
                if (box.width <= 0.0F || box.height <= 0.0F)
                {
                    continue;
                }

                Candidate candidate{};
                candidate.box      = box;
                candidate.score    = best_score;
                candidate.class_id = best_class;
                candidates.push_back(candidate);
            }

            return candidates;
        }

    }  // namespace

    std::vector<cvkit::core::Detection> postprocess_yolo_detections(
        const RawTensorMap&             outputs,
        const LetterboxResult&          preprocess,
        const cvkit::core::Frame&       frame,
        const std::vector<std::string>& labels,
        float                           confidence_threshold,
        float                           iou_threshold)
    {
        std::vector<Candidate> candidates;
        for (const auto& output : outputs)
        {
            auto parsed = parse_yolo_output_tensor(output, preprocess, frame, confidence_threshold);
            candidates.insert(candidates.end(), parsed.begin(), parsed.end());
        }

        return non_maximum_suppression(std::move(candidates), labels, iou_threshold);
    }

}  // namespace cvkit::infer::detail
