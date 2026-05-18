#pragma once

#include "cvkit/infer/device.h"
#include "cvkit/infer/task_io.h"
#include "cvkit/core/types.h"

#include "../../../backends/backend_session.h"
#include "yolo_preprocess_cpu.h"
#include "yolo_preprocess_cuda.h"

#include <string>
#include <cstdint>
#include <vector>

namespace cvkit::infer::detail
{

    enum class YoloPreprocessPath : std::uint8_t
    {
        unsupported,
        host_cpu,
        cuda_device,
    };

    enum class YoloPreprocessStatus : std::uint8_t
    {
        ok,
        invalid_input,
        unsupported_device_path,
    };

    struct YoloPreprocessSource
    {
        const cvkit::infer::ImageValue* image{nullptr};
        YoloPreprocessPath              path{YoloPreprocessPath::unsupported};
        const cvkit::core::Frame*       frame{nullptr};
        MemoryDevice                    memory_device{MemoryDevice::host};
        DeviceRef                       device{};
    };

    struct YoloPreprocessOutcome
    {
        YoloPreprocessStatus status{YoloPreprocessStatus::invalid_input};
        YoloPreprocessSource source{};
        LetterboxResult      result{};
        std::string          error{};

        [[nodiscard]] bool   ok() const
        {
            return status == YoloPreprocessStatus::ok && !result.tensor.data.empty();
        }
    };

    [[nodiscard]] inline YoloPreprocessSource select_yolo_preprocess_source(const TaskInput& input)
    {
        if (const auto* image = input.find<cvkit::infer::ImageValue>("image"); image != nullptr)
        {
            YoloPreprocessSource source{};
            source.image         = image;
            source.memory_device = image->memory_device;
            source.device        = image->device;
            if (image->has_valid_host_layout())
            {
                source.path  = YoloPreprocessPath::host_cpu;
                source.frame = &image->frame;
                return source;
            }

            if (image->memory_device == MemoryDevice::cuda && image->has_valid_device_view())
            {
                source.path = YoloPreprocessPath::cuda_device;
                return source;
            }

            return source;
        }

        if (const auto* frame = input.find<cvkit::core::Frame>("image"); frame != nullptr)
        {
            return YoloPreprocessSource{
                .path          = YoloPreprocessPath::host_cpu,
                .frame         = frame,
                .memory_device = MemoryDevice::host,
                .device        = {},
            };
        }

        return {};
    }

    [[nodiscard]] inline YoloPreprocessOutcome preprocess_yolo(
        const TaskInput&                 input,
        const std::vector<std::int64_t>& input_shape,
        bool                             prefer_device_tensor_output = false)
    {
        YoloPreprocessOutcome outcome{};
        outcome.source = select_yolo_preprocess_source(input);
        switch (outcome.source.path)
        {
            case YoloPreprocessPath::host_cpu:
                if (outcome.source.frame == nullptr)
                {
                    outcome.error = "host_cpu preprocess selected without a valid frame";
                    return outcome;
                }
                outcome.result = preprocess_yolo_cpu(*outcome.source.frame, input_shape);
                if (outcome.result.tensor.data.empty())
                {
                    outcome.error = "host_cpu preprocess failed to produce a tensor";
                    return outcome;
                }
                outcome.status = YoloPreprocessStatus::ok;
                return outcome;
            case YoloPreprocessPath::cuda_device:
            {
                if (outcome.source.image == nullptr)
                {
                    outcome.status = YoloPreprocessStatus::unsupported_device_path;
                    outcome.error  = "cuda_device preprocess selected without a valid image input";
                    return outcome;
                }
                auto result = preprocess_yolo_cuda(
                    *outcome.source.image,
                    input_shape,
                    prefer_device_tensor_output,
                    &outcome.error);
                if (!result.has_value())
                {
                    outcome.status = YoloPreprocessStatus::unsupported_device_path;
                    return outcome;
                }
                outcome.result = std::move(*result);
                outcome.status = YoloPreprocessStatus::ok;
                return outcome;
            }
            case YoloPreprocessPath::unsupported:
            default:
                outcome.error = "no supported YOLO preprocess input source was found";
                return outcome;
        }
    }

}  // namespace cvkit::infer::detail
