#include "face_detection_pipeline.h"

#include "cvkit/infer/tasks/face_detection.h"

#include "../../utils/image_preprocess.h"
#include "../../utils/nms.h"
#include "../../utils/opencv_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        struct ScrfdInput
        {
            RawTensor tensor{};
            float     det_scale_x{1.0F};
            float     det_scale_y{1.0F};
            int       input_width{0};
            int       input_height{0};
        };

        struct ScrfdPreprocessOptions
        {
            bool keep_aspect_ratio{true};
            bool raw_bgr_255{false};
        };

        [[nodiscard]] bool is_scrfd_family(std::string_view family)
        {
            return family.empty() ||
                   family == "scrfd" ||
                   family == "scrfd_raw_bgr";
        }

        [[nodiscard]] ScrfdPreprocessOptions scrfd_preprocess_options(std::string_view family)
        {
            ScrfdPreprocessOptions options{};
            options.keep_aspect_ratio = family != "scrfd_raw_bgr";
            options.raw_bgr_255       = family == "scrfd_raw_bgr";
            return options;
        }

        [[nodiscard]] std::optional<ScrfdInput> build_scrfd_input(
            const cvkit::core::Frame&        frame,
            const std::vector<std::int64_t>& input_shape,
            const ScrfdPreprocessOptions&    options)
        {
            if (input_shape.size() != 4 || input_shape[1] != 3 || input_shape[2] <= 0 || input_shape[3] <= 0)
            {
                return std::nullopt;
            }

            auto source = frame_to_mat_copy(frame);
            if (source.empty() || frame.desc.width <= 0 || frame.desc.height <= 0)
            {
                return std::nullopt;
            }

            const auto input_height = static_cast<int>(input_shape[2]);
            const auto input_width  = static_cast<int>(input_shape[3]);
            int resized_width  = input_width;
            int resized_height = input_height;
            if (options.keep_aspect_ratio)
            {
                const auto image_ratio = static_cast<float>(frame.desc.height) / static_cast<float>(frame.desc.width);
                const auto model_ratio = static_cast<float>(input_height) / static_cast<float>(input_width);
                if (image_ratio > model_ratio)
                {
                    resized_height = input_height;
                    resized_width  = static_cast<int>(static_cast<float>(resized_height) / image_ratio);
                }
                else
                {
                    resized_width  = input_width;
                    resized_height = static_cast<int>(static_cast<float>(resized_width) * image_ratio);
                }
            }
            resized_width  = std::max(1, resized_width);
            resized_height = std::max(1, resized_height);

            cv::Mat resized;
            cv::resize(source, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat canvas(input_height, input_width, CV_8UC3, cv::Scalar(0, 0, 0));
            resized.copyTo(canvas(cv::Rect(0, 0, resized_width, resized_height)));

            cv::Mat model_pixels;
            if (options.raw_bgr_255 && frame.desc.format == cvkit::core::PixelFormat::rgb8)
            {
                cv::cvtColor(canvas, model_pixels, cv::COLOR_RGB2BGR);
            }
            else if (options.raw_bgr_255 || frame.desc.format == cvkit::core::PixelFormat::rgb8)
            {
                model_pixels = canvas;
            }
            else
            {
                cv::cvtColor(canvas, model_pixels, cv::COLOR_BGR2RGB);
            }

            cv::Mat normalized;
            if (options.raw_bgr_255)
            {
                model_pixels.convertTo(normalized, CV_32FC3);
            }
            else
            {
                model_pixels.convertTo(normalized, CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);
            }

            const auto plane_size = static_cast<std::size_t>(input_width * input_height);
            RawTensor  tensor{};
            tensor.name  = "input.1";
            tensor.shape = input_shape;
            tensor.data.resize(plane_size * 3U);

            for (int y = 0; y < input_height; ++y)
            {
                for (int x = 0; x < input_width; ++x)
                {
                    const auto pixel                       = normalized.at<cv::Vec3f>(y, x);
                    const auto offset                      = static_cast<std::size_t>(y * input_width + x);
                    tensor.data[offset]                    = pixel[0];
                    tensor.data[plane_size + offset]       = pixel[1];
                    tensor.data[(2 * plane_size) + offset] = pixel[2];
                }
            }

            ScrfdInput result{};
            result.tensor       = std::move(tensor);
            result.det_scale_x  = static_cast<float>(resized_width) / static_cast<float>(frame.desc.width);
            result.det_scale_y  = static_cast<float>(resized_height) / static_cast<float>(frame.desc.height);
            result.input_width  = input_width;
            result.input_height = input_height;
            return result;
        }

        [[nodiscard]] std::vector<std::int64_t> resolve_scrfd_input_shape(
            const IBackendSession&    backend,
            const cvkit::core::Frame& frame)
        {
            std::vector<std::int64_t> shape{};
            if (const auto* info = backend.input_info(0); info != nullptr && info->shape.size() == 4)
            {
                shape = info->shape;
            }
            else
            {
                shape = {1, 3, 640, 640};
            }
            shape[0] = shape[0] > 0 ? shape[0] : 1;
            shape[1] = 3;
            if (shape[2] <= 0 || shape[3] <= 0)
            {
                shape[2] = 640;
                shape[3] = 640;
            }
            return shape;
        }

        [[nodiscard]] cvkit::core::BBox decode_face_box(
            float                     a,
            float                     b,
            float                     c,
            float                     d,
            const cvkit::core::Frame& frame)
        {
            if (std::max({a, b, c, d}) <= 2.0F)
            {
                a *= static_cast<float>(frame.desc.width);
                c *= static_cast<float>(frame.desc.width);
                b *= static_cast<float>(frame.desc.height);
                d *= static_cast<float>(frame.desc.height);
            }

            float x0 = a;
            float y0 = b;
            float x1 = c;
            float y1 = d;
            if (c <= a || d <= b)
            {
                x0 = a - c * 0.5F;
                y0 = b - d * 0.5F;
                x1 = a + c * 0.5F;
                y1 = b + d * 0.5F;
            }

            cvkit::core::BBox box{};
            box.x             = std::clamp(x0, 0.0F, static_cast<float>(frame.desc.width));
            box.y             = std::clamp(y0, 0.0F, static_cast<float>(frame.desc.height));
            const auto right  = std::clamp(x1, 0.0F, static_cast<float>(frame.desc.width));
            const auto bottom = std::clamp(y1, 0.0F, static_cast<float>(frame.desc.height));
            box.width         = std::max(0.0F, right - box.x);
            box.height        = std::max(0.0F, bottom - box.y);
            return box;
        }

        [[nodiscard]] cvkit::core::BBox decode_scrfd_box(
            float                     center_x,
            float                     center_y,
            float                     left,
            float                     top,
            float                     right,
            float                     bottom,
            float                     det_scale_x,
            float                     det_scale_y,
            const cvkit::core::Frame& frame)
        {
            const auto x0 = (center_x - left) / det_scale_x;
            const auto y0 = (center_y - top) / det_scale_y;
            const auto x1 = (center_x + right) / det_scale_x;
            const auto y1 = (center_y + bottom) / det_scale_y;

            cvkit::core::BBox box{};
            box.x                    = std::clamp(x0, 0.0F, static_cast<float>(frame.desc.width));
            box.y                    = std::clamp(y0, 0.0F, static_cast<float>(frame.desc.height));
            const auto clamped_right = std::clamp(x1, 0.0F, static_cast<float>(frame.desc.width));
            const auto clamped_bottom = std::clamp(y1, 0.0F, static_cast<float>(frame.desc.height));
            box.width                = std::max(0.0F, clamped_right - box.x);
            box.height               = std::max(0.0F, clamped_bottom - box.y);
            return box;
        }

        void append_keypoints(
            Candidate&                 candidate,
            const std::function<float(std::size_t, std::size_t)>& access,
            std::size_t                row_index,
            std::size_t                attrs,
            const cvkit::core::Frame&  frame)
        {
            constexpr std::size_t kLandmarkOffset = 5U;
            if (attrs < kLandmarkOffset + 10U)
            {
                return;
            }

            candidate.keypoints.reserve((attrs - kLandmarkOffset) / 2U);
            for (std::size_t offset = kLandmarkOffset; offset + 1U < attrs; offset += 2U)
            {
                auto x = access(row_index, offset);
                auto y = access(row_index, offset + 1U);
                if (std::max(x, y) <= 2.0F)
                {
                    x *= static_cast<float>(frame.desc.width);
                    y *= static_cast<float>(frame.desc.height);
                }
                candidate.keypoints.push_back(cvkit::core::Point2f{
                    std::clamp(x, 0.0F, static_cast<float>(frame.desc.width)),
                    std::clamp(y, 0.0F, static_cast<float>(frame.desc.height))});
            }
        }

        [[nodiscard]] std::size_t tensor_rows(const RawTensor& tensor)
        {
            if (tensor.shape.size() == 1)
            {
                return static_cast<std::size_t>(std::max<std::int64_t>(0, tensor.shape[0]));
            }
            if (tensor.shape.size() == 3)
            {
                return static_cast<std::size_t>(std::max<std::int64_t>(0, tensor.shape[1]));
            }
            if (tensor.shape.size() == 2)
            {
                return static_cast<std::size_t>(std::max<std::int64_t>(0, tensor.shape[0]));
            }
            return 0U;
        }

        [[nodiscard]] std::size_t tensor_cols(const RawTensor& tensor)
        {
            if (tensor.shape.size() == 1)
            {
                return 1U;
            }
            if (tensor.shape.size() == 3)
            {
                return static_cast<std::size_t>(std::max<std::int64_t>(0, tensor.shape[2]));
            }
            if (tensor.shape.size() == 2)
            {
                return static_cast<std::size_t>(std::max<std::int64_t>(0, tensor.shape[1]));
            }
            return 0U;
        }

        [[nodiscard]] float tensor_value(const RawTensor& tensor, std::size_t row, std::size_t col)
        {
            const auto cols  = tensor_cols(tensor);
            const auto index = row * cols + col;
            return index < tensor.data.size() ? tensor.data[index] : 0.0F;
        }

        [[nodiscard]] std::vector<Candidate> parse_scrfd_outputs(
            const RawTensorMap&       outputs,
            const ScrfdInput&         preprocess,
            const cvkit::core::Frame& frame,
            float                     confidence_threshold)
        {
            std::vector<Candidate> candidates;
            const auto output_count = outputs.size();
            if (output_count != 6U && output_count != 9U && output_count != 10U && output_count != 15U)
            {
                return candidates;
            }

            const bool        use_kps     = output_count == 9U || output_count == 15U;
            const std::size_t fmc         = output_count == 10U || output_count == 15U ? 5U : 3U;
            const int         num_anchors = fmc == 3U ? 2 : 1;
            const int         strides[5]  = {8, 16, 32, 64, 128};

            for (std::size_t level = 0; level < fmc; ++level)
            {
                const auto& scores = outputs[level];
                const auto& boxes  = outputs[level + fmc];
                const auto* kps    = use_kps ? &outputs[level + fmc * 2U] : nullptr;
                const auto  stride = strides[level];
                const auto  rows   = tensor_rows(scores);
                if (rows == 0U || tensor_rows(boxes) != rows || tensor_cols(boxes) < 4U)
                {
                    continue;
                }
                if (kps != nullptr && (tensor_rows(*kps) != rows || tensor_cols(*kps) < 10U))
                {
                    continue;
                }

                const auto feature_width = std::max(1, preprocess.input_width / stride);
                candidates.reserve(candidates.size() + rows);
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const auto score = tensor_value(scores, row, 0);
                    if (score < confidence_threshold)
                    {
                        continue;
                    }

                    const auto anchor_index = row / static_cast<std::size_t>(num_anchors);
                    const auto grid_x       = static_cast<int>(anchor_index % static_cast<std::size_t>(feature_width));
                    const auto grid_y       = static_cast<int>(anchor_index / static_cast<std::size_t>(feature_width));
                    const auto center_x      = static_cast<float>(grid_x * stride);
                    const auto center_y      = static_cast<float>(grid_y * stride);

                    Candidate candidate{};
                    candidate.box = decode_scrfd_box(
                        center_x,
                        center_y,
                        tensor_value(boxes, row, 0) * static_cast<float>(stride),
                        tensor_value(boxes, row, 1) * static_cast<float>(stride),
                        tensor_value(boxes, row, 2) * static_cast<float>(stride),
                        tensor_value(boxes, row, 3) * static_cast<float>(stride),
                        preprocess.det_scale_x,
                        preprocess.det_scale_y,
                        frame);
                    if (candidate.box.width <= 0.0F || candidate.box.height <= 0.0F)
                    {
                        continue;
                    }

                    candidate.score    = score;
                    candidate.class_id = 0;
                    if (kps != nullptr)
                    {
                        candidate.keypoints.reserve(5U);
                        for (std::size_t index = 0; index < 5U; ++index)
                        {
                            const auto x = (center_x + tensor_value(*kps, row, index * 2U) * static_cast<float>(stride)) / preprocess.det_scale_x;
                            const auto y = (center_y + tensor_value(*kps, row, index * 2U + 1U) * static_cast<float>(stride)) / preprocess.det_scale_y;
                            candidate.keypoints.push_back(cvkit::core::Point2f{
                                std::clamp(x, 0.0F, static_cast<float>(frame.desc.width)),
                                std::clamp(y, 0.0F, static_cast<float>(frame.desc.height))});
                        }
                    }
                    candidates.push_back(std::move(candidate));
                }
            }

            return candidates;
        }

        [[nodiscard]] std::vector<Candidate> parse_face_detection_output_tensor(
            const RawTensor&          output,
            const cvkit::core::Frame& frame,
            float                     confidence_threshold)
        {
            std::vector<Candidate> candidates;
            if (output.data.empty())
            {
                return candidates;
            }

            const auto& shape = output.shape;
            const auto* data  = output.data.data();
            std::size_t boxes = 0;
            std::size_t attrs = 0;
            bool        transposed = false;

            if (shape.size() == 3)
            {
                const auto dim1 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[1]));
                const auto dim2 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[2]));
                if (dim1 >= 5 && dim2 >= 5)
                {
                    boxes      = dim1;
                    attrs      = dim2;
                    transposed = false;
                }
                else if (dim1 >= 5)
                {
                    attrs      = dim1;
                    boxes      = dim2;
                    transposed = true;
                }
                else if (dim2 >= 5)
                {
                    boxes      = dim1;
                    attrs      = dim2;
                    transposed = false;
                }
            }
            else if (shape.size() == 2)
            {
                const auto dim0 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[0]));
                const auto dim1 = static_cast<std::size_t>(std::max<std::int64_t>(1, shape[1]));
                if (dim0 >= 5 && dim1 >= 5)
                {
                    attrs      = dim0 <= dim1 ? dim0 : dim1;
                    boxes      = dim0 <= dim1 ? dim1 : dim0;
                    transposed = dim0 <= dim1;
                }
            }

            if (boxes == 0U || attrs < 5U || output.data.size() < boxes * attrs)
            {
                return candidates;
            }

            std::function<float(std::size_t, std::size_t)> access = [&](std::size_t box_index, std::size_t attr_index) -> float
            {
                if (transposed)
                {
                    return data[attr_index * boxes + box_index];
                }
                return data[box_index * attrs + attr_index];
            };

            candidates.reserve(boxes);
            for (std::size_t box_index = 0; box_index < boxes; ++box_index)
            {
                const auto score = access(box_index, 4);
                if (score < confidence_threshold)
                {
                    continue;
                }

                auto box = decode_face_box(
                    access(box_index, 0),
                    access(box_index, 1),
                    access(box_index, 2),
                    access(box_index, 3),
                    frame);
                if (box.width <= 0.0F || box.height <= 0.0F)
                {
                    continue;
                }

                Candidate candidate{};
                candidate.box      = box;
                candidate.score    = score;
                candidate.class_id = 0;
                append_keypoints(candidate, access, box_index, attrs, frame);
                candidates.push_back(std::move(candidate));
            }

            return candidates;
        }

    }  // namespace

    TaskKind FaceDetectionPipeline::task() const
    {
        return TaskKind::face_detection;
    }

    TaskSchema FaceDetectionPipeline::schema() const
    {
        return face_detection_schema();
    }

    TaskOutput FaceDetectionPipeline::run_sync(
        const IBackendSession& backend,
        const TaskInput&       input,
        const PipelineContext& context) const
    {
        const auto frame = resolve_host_image_input(input);
        if (!frame.has_value())
        {
            return {};
        }

        const auto scrfd = is_scrfd_family(context.spec.family);
        const auto input_shape = scrfd
            ? resolve_scrfd_input_shape(backend, *frame)
            : resolve_nchw_input_shape(backend, *frame);

        const auto preprocess_options = scrfd_preprocess_options(context.spec.family);
        auto       model_input        = build_scrfd_input(*frame, input_shape, preprocess_options);
        if (!model_input.has_value() || model_input->tensor.data.empty())
        {
            return {};
        }

        RawTensorMap inputs{};
        inputs.push_back(model_input->tensor);
        const auto outputs = backend.run(inputs);
        if (outputs.empty())
        {
            return {};
        }

        auto candidates = scrfd
            ? parse_scrfd_outputs(outputs, *model_input, *frame, context.confidence_threshold)
            : std::vector<Candidate>{};
        if (!scrfd)
        {
            for (const auto& output_tensor : outputs)
            {
                auto parsed = parse_face_detection_output_tensor(output_tensor, *frame, context.confidence_threshold);
                candidates.insert(
                    candidates.end(),
                    std::make_move_iterator(parsed.begin()),
                    std::make_move_iterator(parsed.end()));
            }
        }

        TaskOutput output{};
        output.add("detections", non_maximum_suppression(std::move(candidates), context.labels, context.iou_threshold));
        return output;
    }

}  // namespace cvkit::infer::detail
