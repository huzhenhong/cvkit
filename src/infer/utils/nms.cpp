#include "nms.h"

#include "bbox.h"
#include "labels.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{

    std::vector<cvkit::core::Detection> non_maximum_suppression(
        std::vector<Candidate>          candidates,
        const std::vector<std::string>& labels,
        float                           iou_threshold)
    {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs)
            { return lhs.score > rhs.score; });

        std::vector<cvkit::core::Detection> detections;
        std::vector<bool>                   suppressed(candidates.size(), false);
        detections.reserve(candidates.size());

        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            if (suppressed[i])
            {
                continue;
            }

            const auto&            keep = candidates[i];
            cvkit::core::Detection detection{};
            detection.box      = keep.box;
            detection.score    = keep.score;
            detection.class_id = keep.class_id;
            detection.label    = resolve_label(labels, keep.class_id);
            detection.keypoints = keep.keypoints;
            detections.push_back(std::move(detection));

            for (std::size_t j = i + 1; j < candidates.size(); ++j)
            {
                if (suppressed[j] || candidates[j].class_id != keep.class_id)
                {
                    continue;
                }

                if (intersection_over_union(keep.box, candidates[j].box) >= iou_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return detections;
    }

}  // namespace cvkit::infer::detail
