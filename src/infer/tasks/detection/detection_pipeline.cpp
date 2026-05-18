#include "detection_pipeline.h"

#include "cvkit/infer/tasks/detection.h"

#include "yolo/yolo_postprocess.h"
#include "yolo/yolo_preprocess.h"

#include "../../utils/task_input_utils.h"
#include "../../utils/tensor_layout.h"

#include <optional>

#if defined(CVKIT_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

namespace cvkit::infer::detail
{

    namespace
    {

        struct DetectionInferenceRequest
        {
            LetterboxResult preprocess{};
            RawTensorMap    inputs{};
        };

        [[nodiscard]] std::optional<RawTensorMap> materialize_host_outputs(const RawTensorMap& outputs)
        {
            RawTensorMap host_outputs;
            host_outputs.reserve(outputs.size());

            for (const auto& output : outputs)
            {
                if (output.memory_device == MemoryDevice::host)
                {
                    host_outputs.push_back(output);
                    continue;
                }

                if (output.memory_device != MemoryDevice::cuda || output.data_type != TensorDataType::float32 || !output.has_valid_device_view())
                {
                    return std::nullopt;
                }

#if !defined(CVKIT_WITH_CUDA_RUNTIME)
                return std::nullopt;
#else
                RawTensor host_output{};
                host_output.name          = output.name;
                host_output.shape         = output.shape;
                host_output.data_type     = output.data_type;
                host_output.memory_device = MemoryDevice::host;
                host_output.storage       = StorageKind::owned;
                host_output.data.resize(output.element_count());

                if (cudaMemcpy(
                        host_output.data.data(),
                        output.external_data,
                        output.packed_byte_size(),
                        cudaMemcpyDeviceToHost) != cudaSuccess)
                {
                    return std::nullopt;
                }

                host_outputs.push_back(std::move(host_output));
#endif
            }

            return host_outputs;
        }

        [[nodiscard]] std::optional<DetectionInferenceRequest> build_detection_request(
            const IBackendSession& backend,
            const TaskInput&       input,
            std::string*           error_message = nullptr)
        {
            const auto source = select_yolo_preprocess_source(input);
            if (source.path == YoloPreprocessPath::unsupported)
            {
                if (error_message != nullptr)
                {
                    *error_message = "no supported image input is available for detection preprocess";
                }
                return std::nullopt;
            }
            if (source.path == YoloPreprocessPath::host_cpu && source.frame == nullptr)
            {
                if (error_message != nullptr)
                {
                    *error_message = "host_cpu preprocess selected without a valid frame";
                }
                return std::nullopt;
            }

            std::vector<std::int64_t> backend_shape{};
            if (const auto* input_info = backend.input_info(0); input_info != nullptr)
            {
                backend_shape = input_info->shape;
            }

            if (backend_shape.empty())
            {
                const auto* frame = source.frame;
                if (frame == nullptr && source.image != nullptr)
                {
                    frame = &source.image->frame;
                }
                if (frame != nullptr)
                {
                    backend_shape = {1, 3, frame->desc.height, frame->desc.width};
                }
            }

            auto        resolved_shape = backend_shape;
            const auto* layout_frame   = source.frame;
            if (layout_frame == nullptr && source.image != nullptr)
            {
                layout_frame = &source.image->frame;
            }
            if (!backend_shape.empty() && layout_frame != nullptr)
            {
                resolved_shape = resolve_input_shape(backend_shape, *layout_frame);
            }
            const bool prefer_device_tensor_output =
                source.path == YoloPreprocessPath::cuda_device && backend.backend() == Backend::tensorrt;
            auto preprocess = preprocess_yolo(input, resolved_shape, prefer_device_tensor_output);
            if (!preprocess.ok())
            {
                if (error_message != nullptr)
                {
                    *error_message = preprocess.error;
                }
                return std::nullopt;
            }

            DetectionInferenceRequest request{};
            request.inputs.push_back(preprocess.result.tensor);
            request.preprocess = std::move(preprocess.result);
            return request;
        }

    }  // namespace

    bool prepare_detection_inference(
        const IBackendSession& backend,
        const TaskInput&       input,
        Packet&                packet)
    {
        std::string error_message{};
        auto        request = build_detection_request(backend, input, &error_message);
        if (!request.has_value())
        {
            if (!error_message.empty())
            {
                packet.put("detection.preprocess_error", std::move(error_message));
            }
            return false;
        }

        packet.put("detection.preprocess", std::move(request->preprocess));
        packet.put("detection.raw_outputs", backend.run(request->inputs));
        return true;
    }

    bool prepare_detection_inference_async(
        const IBackendSession& backend,
        const TaskInput&       input,
        Packet&                packet,
        BackendFuture&         future)
    {
        std::string error_message{};
        auto        request = build_detection_request(backend, input, &error_message);
        if (!request.has_value())
        {
            if (!error_message.empty())
            {
                packet.put("detection.preprocess_error", std::move(error_message));
            }
            return false;
        }

        packet.put("detection.preprocess", std::move(request->preprocess));
        future = backend.run_async(request->inputs);
        return future.valid();
    }

    std::vector<cvkit::core::Detection> finalize_detection_output(
        const Packet&          packet,
        const PipelineContext& context)
    {
        const auto* frame      = find_host_frame_input(packet.input);
        const auto* preprocess = packet.get<LetterboxResult>("detection.preprocess");
        const auto* outputs    = packet.get<RawTensorMap>("detection.raw_outputs");
        if (frame == nullptr || preprocess == nullptr || outputs == nullptr)
        {
            return {};
        }

        auto host_outputs = materialize_host_outputs(*outputs);
        if (!host_outputs.has_value())
        {
            return {};
        }

        return postprocess_yolo_detections(
            *host_outputs,
            *preprocess,
            *frame,
            context.labels,
            context.confidence_threshold,
            context.iou_threshold);
    }

    TaskKind DetectionPipeline::task() const
    {
        return TaskKind::detection;
    }

    TaskSchema DetectionPipeline::schema() const
    {
        return detection_schema();
    }

    TaskOutput DetectionPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        Packet packet{};
        packet.input = input;

        if (!prepare_detection_inference(backend, input, packet))
        {
            return {};
        }

        TaskOutput output{};
        output.add("detections", finalize_detection_output(packet, context));
        return output;
    }

}  // namespace cvkit::infer::detail
