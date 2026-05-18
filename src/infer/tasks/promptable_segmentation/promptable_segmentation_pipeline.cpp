#include "promptable_segmentation_pipeline.h"

#include "cvkit/infer/tasks/promptable_segmentation.h"

#include "../../backends/backend_session.h"
#include "../../utils/image_value_utils.h"
#include "../../utils/opencv_utils.h"
#include "../../utils/task_input_utils.h"
#include "../../utils/tensor_value_utils.h"
#include "promptable_preprocess_cuda.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        constexpr int   kEncoderImageSize = 1024;
        constexpr int   kDecoderMaskSize  = 256;
        constexpr float kSamMeanR         = 123.675F;
        constexpr float kSamMeanG         = 116.28F;
        constexpr float kSamMeanB         = 103.53F;
        constexpr float kSamStdR          = 58.395F;
        constexpr float kSamStdG          = 57.12F;
        constexpr float kSamStdB          = 57.375F;

        struct PromptablePreprocessResult
        {
            int                width{0};
            int                height{0};
            std::vector<float> point_coords{};
            std::vector<float> point_labels{};
        };

        enum class PromptableFamily : std::uint8_t
        {
            invalid,
            encoder,
            decoder,
            combined,
        };

        [[nodiscard]] RawTensor build_encoder_input(const cvkit::core::Frame& frame)
        {
            RawTensor tensor{};
            tensor.name  = "batched_images";
            tensor.shape = {1, 3, kEncoderImageSize, kEncoderImageSize};

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
            cv::resize(rgb, resized, cv::Size(kEncoderImageSize, kEncoderImageSize), 0.0, 0.0, cv::INTER_LINEAR);
            resized.convertTo(resized, CV_32FC3);

            tensor.data.resize(static_cast<std::size_t>(3 * kEncoderImageSize * kEncoderImageSize));
            const auto plane_size = static_cast<std::size_t>(kEncoderImageSize * kEncoderImageSize);
            for (int y = 0; y < kEncoderImageSize; ++y)
            {
                for (int x = 0; x < kEncoderImageSize; ++x)
                {
                    const auto pixel                       = resized.at<cv::Vec3f>(y, x);
                    const auto offset                      = static_cast<std::size_t>(y * kEncoderImageSize + x);
                    tensor.data[offset]                    = (pixel[0] - kSamMeanR) / kSamStdR;
                    tensor.data[plane_size + offset]       = (pixel[1] - kSamMeanG) / kSamStdG;
                    tensor.data[(2 * plane_size) + offset] = (pixel[2] - kSamMeanB) / kSamStdB;
                }
            }

            return tensor;
        }

        [[nodiscard]] PromptablePreprocessResult build_promptable_prompts(
            const TaskInput&          input,
            const cvkit::core::Frame& frame)
        {
            PromptablePreprocessResult result{};
            result.width  = std::max(1, frame.desc.width);
            result.height = std::max(1, frame.desc.height);

            auto append_point = [&](float x, float y, float label)
            {
                const auto scaled_x =
                    std::clamp(x * static_cast<float>(kEncoderImageSize) / static_cast<float>(result.width), 0.0F, static_cast<float>(kEncoderImageSize - 1));
                const auto scaled_y =
                    std::clamp(y * static_cast<float>(kEncoderImageSize) / static_cast<float>(result.height), 0.0F, static_cast<float>(kEncoderImageSize - 1));
                result.point_coords.push_back(scaled_x);
                result.point_coords.push_back(scaled_y);
                result.point_labels.push_back(label);
            };

            if (const auto* box = input.find<cvkit::core::BBox>("box"); box != nullptr)
            {
                append_point(box->x, box->y, 2.0F);
                append_point(box->x + box->width, box->y + box->height, 3.0F);
            }

            const auto* point_labels = input.find<std::vector<float>>("point_labels");
            if (const auto* points = input.find<std::vector<cvkit::core::Point2f>>("points"); points != nullptr)
            {
                for (std::size_t index = 0; index < points->size(); ++index)
                {
                    const auto& point = (*points)[index];
                    const auto  label =
                        (point_labels != nullptr && index < point_labels->size()) ? (*point_labels)[index] : 1.0F;
                    append_point(point.x, point.y, label);
                }
            }

            if (result.point_labels.empty())
            {
                append_point(
                    static_cast<float>(result.width) * 0.5F,
                    static_cast<float>(result.height) * 0.5F,
                    1.0F);
            }

            return result;
        }

        [[nodiscard]] RawTensor build_decoder_coords(const PromptablePreprocessResult& prompt)
        {
            RawTensor tensor{};
            tensor.name  = "batched_point_coords";
            tensor.shape = {1, 1, static_cast<std::int64_t>(prompt.point_labels.size()), 2};
            tensor.data  = prompt.point_coords;
            return tensor;
        }

        [[nodiscard]] RawTensor build_decoder_labels(const PromptablePreprocessResult& prompt)
        {
            RawTensor tensor{};
            tensor.name  = "batched_point_labels";
            tensor.shape = {1, 1, static_cast<std::int64_t>(prompt.point_labels.size())};
            tensor.data  = prompt.point_labels;
            return tensor;
        }

        [[nodiscard]] cvkit::core::Frame build_mask_frame(
            const std::vector<float>& mask_logits,
            int                       source_width,
            int                       source_height)
        {
            cvkit::core::Frame mask{};
            if (source_width <= 0 || source_height <= 0 || mask_logits.empty())
            {
                return mask;
            }

            cv::Mat logits(kDecoderMaskSize, kDecoderMaskSize, CV_32FC1, const_cast<float*>(mask_logits.data()));
            cv::Mat resized;
            cv::resize(logits, resized, cv::Size(source_width, source_height), 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat binary;
            cv::threshold(resized, binary, 0.0, 255.0, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8UC1);

            mask.desc.width    = source_width;
            mask.desc.height   = source_height;
            mask.desc.channels = 1;
            mask.desc.format   = cvkit::core::PixelFormat::unknown;
            mask.data.assign(binary.data, binary.data + (binary.total() * binary.elemSize()));
            return mask;
        }

        [[nodiscard]] bool is_encoder_family(std::string_view family)
        {
            return family == "efficient_sam_encoder";
        }

        [[nodiscard]] bool is_decoder_family(std::string_view family)
        {
            return family == "efficient_sam_decoder";
        }

        [[nodiscard]] bool is_combined_family(std::string_view family)
        {
            return family == "efficient_sam";
        }

        [[nodiscard]] PromptableFamily parse_promptable_family(std::string_view family)
        {
            if (is_encoder_family(family))
            {
                return PromptableFamily::encoder;
            }
            if (is_decoder_family(family))
            {
                return PromptableFamily::decoder;
            }
            if (is_combined_family(family))
            {
                return PromptableFamily::combined;
            }
            return PromptableFamily::invalid;
        }

        [[nodiscard]] bool requires_image(PromptableFamily family)
        {
            return family == PromptableFamily::encoder || family == PromptableFamily::combined;
        }

        [[nodiscard]] bool requires_external_embeddings(PromptableFamily family)
        {
            return family == PromptableFamily::decoder;
        }

        [[nodiscard]] bool requires_decoder_model(
            PromptableFamily family,
            const ModelSpec& spec)
        {
            return family == PromptableFamily::decoder ||
                   (family == PromptableFamily::combined && !spec.aux_model_path.empty());
        }

        [[nodiscard]] cvkit::infer::TensorValue make_selected_logits_tensor(const std::vector<float>& logits)
        {
            return cvkit::infer::TensorValue{
                "logits",
                {1, kDecoderMaskSize, kDecoderMaskSize},
                logits};
        }

        [[nodiscard]] std::optional<RawTensor> run_encoder_model(
            const IBackendSession&          backend,
            const cvkit::infer::ImageValue* image,
            const cvkit::core::Frame&       frame,
            const PipelineContext&          context,
            Packet&                         packet)
        {
            RawTensor  encoder_input{};
            const bool prefer_device_tensor_input =
                image != nullptr &&
                image->memory_device == cvkit::infer::MemoryDevice::cuda &&
                context.spec.backend == Backend::onnxruntime &&
                context.spec.device.kind == DeviceKind::cuda;

            if (prefer_device_tensor_input)
            {
                std::string preprocess_error{};
                auto        cuda_input = preprocess_promptable_encoder_cuda(
                    *image,
                    true,
                    &preprocess_error);
                if (!cuda_input.has_value())
                {
                    if (!preprocess_error.empty())
                    {
                        packet.put("promptable.error", std::move(preprocess_error));
                    }
                    return std::nullopt;
                }
                encoder_input = std::move(*cuda_input);
            }
            else
            {
                encoder_input = build_encoder_input(frame);
                if (encoder_input.data.empty())
                {
                    return std::nullopt;
                }
            }

            RawTensorMap encoder_inputs{};
            encoder_inputs.push_back(std::move(encoder_input));
            auto encoder_outputs = backend.run(encoder_inputs);
            if (encoder_outputs.empty())
            {
                return std::nullopt;
            }

            auto encoder_embedding = std::move(encoder_outputs.front());
            packet.put("promptable.image_embeddings", encoder_embedding);
            packet.put("promptable.ready", true);
            return encoder_embedding;
        }

        [[nodiscard]] bool resolve_decoder_embedding(
            PromptableFamily         family,
            const TaskInput&         input,
            std::optional<RawTensor> encoder_embedding,
            RawTensor&               embedding_input,
            std::string*             error_message = nullptr)
        {
            if (requires_external_embeddings(family))
            {
                const auto* embeddings = input.find<cvkit::infer::TensorValue>("image_embeddings");
                if (embeddings == nullptr)
                {
                    return false;
                }
                auto host_embedding = materialize_host_tensor(*embeddings, error_message);
                if (!host_embedding.has_value())
                {
                    return false;
                }
                embedding_input = std::move(*host_embedding);
            }
            else if (encoder_embedding.has_value())
            {
                embedding_input = std::move(*encoder_embedding);
            }
            return !embedding_input.data.empty() && !embedding_input.shape.empty();
        }

        [[nodiscard]] std::unique_ptr<IBackendSession> create_decoder_session(
            PromptableFamily       family,
            const PipelineContext& context)
        {
            auto decoder = create_backend_session(context.spec.backend);
            if (decoder == nullptr)
            {
                return nullptr;
            }

            auto decoder_spec = context.spec;
            if (family != PromptableFamily::decoder)
            {
                decoder_spec.model_path = context.spec.aux_model_path;
            }
            if (!decoder->load(decoder_spec) || !decoder->ready())
            {
                return nullptr;
            }
            return decoder;
        }

        [[nodiscard]] RawTensorMap run_decoder_model(
            const IBackendSession&            decoder,
            RawTensor                         embedding_input,
            const PromptablePreprocessResult& prompt)
        {
            RawTensorMap decoder_inputs{};
            decoder_inputs.push_back(std::move(embedding_input));
            decoder_inputs.push_back(build_decoder_coords(prompt));
            decoder_inputs.push_back(build_decoder_labels(prompt));
            return decoder.run(decoder_inputs);
        }

        [[nodiscard]] bool select_best_mask(
            const RawTensor&    mask_tensor,
            const RawTensor&    iou_tensor,
            std::vector<float>& best_mask,
            FloatListValue&     scores)
        {
            if (mask_tensor.shape.size() != 5 || iou_tensor.data.empty())
            {
                return false;
            }

            const auto mask_candidates = static_cast<std::size_t>(mask_tensor.shape[2]);
            const auto mask_plane_size = static_cast<std::size_t>(mask_tensor.shape[3] * mask_tensor.shape[4]);
            if (mask_candidates == 0 || mask_plane_size == 0)
            {
                return false;
            }

            std::size_t best_index = 0;
            float       best_score = std::numeric_limits<float>::lowest();
            for (std::size_t index = 0; index < iou_tensor.data.size(); ++index)
            {
                if (iou_tensor.data[index] > best_score)
                {
                    best_score = iou_tensor.data[index];
                    best_index = index;
                }
            }
            best_index = std::min(best_index, mask_candidates - 1);

            const auto offset = best_index * mask_plane_size;
            if (offset + mask_plane_size > mask_tensor.data.size())
            {
                return false;
            }

            best_mask.assign(
                mask_tensor.data.begin() + static_cast<std::ptrdiff_t>(offset),
                mask_tensor.data.begin() + static_cast<std::ptrdiff_t>(offset + mask_plane_size));
            scores = iou_tensor.data;
            return true;
        }

        void store_promptable_outputs(
            Packet&                           packet,
            const PromptablePreprocessResult& prompt,
            const RawTensor&                  mask_tensor,
            const std::vector<float>&         best_mask,
            const FloatListValue&             scores)
        {
            packet.put(
                "promptable.mask",
                build_mask_frame(
                    best_mask,
                    prompt.width,
                    prompt.height));
            packet.put("promptable.low_res_masks", mask_tensor);
            packet.put("promptable.logits", make_selected_logits_tensor(best_mask));
            packet.put("promptable.scores", scores);
            packet.put("promptable.ready", true);
        }

        [[nodiscard]] bool run_promptable_segmentation_models(
            const IBackendSession& backend,
            const TaskInput&       input,
            const PipelineContext& context,
            Packet&                packet)
        {
            if (context.spec.backend != Backend::onnxruntime)
            {
                return false;
            }

            const auto family = parse_promptable_family(context.spec.family);
            if (family == PromptableFamily::invalid)
            {
                return false;
            }

            std::optional<cvkit::core::Frame> prompt_frame{};
            const auto*                       image = input.find<cvkit::infer::ImageValue>("image");
            if (image != nullptr)
            {
                std::string image_error{};
                prompt_frame = materialize_host_frame(*image, &image_error);
                if (!prompt_frame.has_value())
                {
                    if (!image_error.empty())
                    {
                        packet.put("promptable.error", std::move(image_error));
                    }
                    return false;
                }
            }
            else if (const auto* frame = input.find<cvkit::core::Frame>("image"); frame != nullptr)
            {
                prompt_frame = *frame;
            }

            if (requires_image(family) && !prompt_frame.has_value())
            {
                packet.put("promptable.error", std::string{"promptable segmentation requires an image input"});
                return false;
            }

            std::optional<RawTensor> encoder_embedding{};
            if (requires_image(family))
            {
                encoder_embedding = run_encoder_model(backend, image, *prompt_frame, context, packet);
                if (!encoder_embedding.has_value())
                {
                    return false;
                }
                if (family == PromptableFamily::encoder)
                {
                    return true;
                }
            }

            RawTensor   embedding_input{};
            std::string embedding_error{};
            if (!resolve_decoder_embedding(
                    family,
                    input,
                    std::move(encoder_embedding),
                    embedding_input,
                    &embedding_error))
            {
                if (!embedding_error.empty())
                {
                    packet.put("promptable.error", std::move(embedding_error));
                }
                return false;
            }

            if (!requires_decoder_model(family, context.spec))
            {
                return false;
            }

            auto decoder = create_decoder_session(family, context);
            if (decoder == nullptr)
            {
                return false;
            }

            const auto prompt =
                build_promptable_prompts(input, prompt_frame.has_value() ? *prompt_frame : cvkit::core::Frame{});
            auto decoder_outputs = run_decoder_model(*decoder, std::move(embedding_input), prompt);
            if (decoder_outputs.size() < 2)
            {
                return false;
            }

            const auto*        mask_tensor = &decoder_outputs[0];
            const auto*        iou_tensor  = &decoder_outputs[1];
            std::vector<float> best_mask{};
            FloatListValue     scores{};
            if (!select_best_mask(*mask_tensor, *iou_tensor, best_mask, scores))
            {
                return false;
            }

            store_promptable_outputs(packet, prompt, *mask_tensor, best_mask, scores);
            return true;
        }

    }  // namespace

    bool prepare_promptable_segmentation_inference(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context,
        Packet&                packet)
    {
        return run_promptable_segmentation_models(backend, input, context, packet);
    }

    TaskOutput finalize_promptable_segmentation_output(
        const Packet&          packet,
        const PipelineContext& context)
    {
        static_cast<void>(context);

        TaskOutput output{};
        if (const auto* ready = packet.get<bool>("promptable.ready"); ready == nullptr || !*ready)
        {
            return output;
        }

        if (const auto* embeddings = packet.get<cvkit::infer::TensorValue>("promptable.image_embeddings");
            embeddings != nullptr)
        {
            output.add("image_embeddings", *embeddings);
        }
        if (const auto* mask = packet.get<cvkit::core::Frame>("promptable.mask"); mask != nullptr)
        {
            output.add("mask", cvkit::infer::MaskValue{*mask});
        }
        if (const auto* low_res_masks = packet.get<cvkit::infer::TensorValue>("promptable.low_res_masks");
            low_res_masks != nullptr)
        {
            output.add("low_res_masks", *low_res_masks);
        }
        if (const auto* logits = packet.get<cvkit::infer::TensorValue>("promptable.logits");
            logits != nullptr)
        {
            output.add("logits", *logits);
        }
        if (const auto* scores = packet.get<std::vector<float>>("promptable.scores"); scores != nullptr)
        {
            output.add("scores", *scores);
        }
        return output;
    }

    TaskKind PromptableSegmentationPipeline::task() const
    {
        return TaskKind::promptable_segmentation;
    }

    TaskSchema PromptableSegmentationPipeline::schema() const
    {
        return promptable_segmentation_schema();
    }

    TaskOutput PromptableSegmentationPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        Packet packet{};
        packet.input = input;
        if (!prepare_promptable_segmentation_inference(backend, input, context, packet))
        {
            return {};
        }
        return finalize_promptable_segmentation_output(packet, context);
    }

}  // namespace cvkit::infer::detail
