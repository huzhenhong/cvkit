#pragma once

#include "cvkit/infer/task_io.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] inline bool has_valid_frame_desc(const cvkit::core::Frame& frame)
    {
        return frame.desc.width > 0 && frame.desc.height > 0 && frame.desc.channels > 0;
    }

    [[nodiscard]] inline const cvkit::core::Frame* find_image_frame_input(const TaskInput& input)
    {
        if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
        {
            return has_valid_frame_desc(image->frame) ? &image->frame : nullptr;
        }

        const auto* frame = input.find<cvkit::core::Frame>("image");
        return frame != nullptr && has_valid_frame_desc(*frame) ? frame : nullptr;
    }

    [[nodiscard]] inline const cvkit::core::Frame* find_host_frame_input(const TaskInput& input)
    {
        if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
        {
            return image->has_valid_host_layout() ? &image->frame : nullptr;
        }
        return input.find<cvkit::core::Frame>("image");
    }

}  // namespace cvkit::infer::detail
