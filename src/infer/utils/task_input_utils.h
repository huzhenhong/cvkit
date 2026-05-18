#pragma once

#include "cvkit/infer/task_io.h"

namespace cvkit::infer::detail
{

    [[nodiscard]] inline const cvkit::core::Frame* find_host_frame_input(const TaskInput& input)
    {
        if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
        {
            return image->has_valid_host_layout() ? &image->frame : nullptr;
        }
        return input.find<cvkit::core::Frame>("image");
    }

}  // namespace cvkit::infer::detail
