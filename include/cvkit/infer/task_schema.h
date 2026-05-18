#pragma once

#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/task.h"

#include <string>
#include <vector>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT ValueKind : unsigned char
    {
        image,
        frame,
        mask,
        classification,
        detections,
        points2f,
        bbox,
        bbox_list,
        keypoints,
        tensor,
        text,
        floats,
    };

    struct BK_INFER_EXPORT IOField
    {
        std::string name{};
        ValueKind   kind{ValueKind::text};
        bool        optional{false};
    };

    struct BK_INFER_EXPORT TaskSchema
    {
        TaskKind             task{TaskKind::unknown};
        std::vector<IOField> inputs{};
        std::vector<IOField> outputs{};
    };

}  // namespace cvkit::infer
