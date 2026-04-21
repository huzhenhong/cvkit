#include "cvkit/infer/model.h"

#if defined(CVKIT_WITH_ONNXRUNTIME)
    #include <onnxruntime_cxx_api.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::infer
{

#if defined(CVKIT_WITH_ONNXRUNTIME)
    namespace
    {

        struct LetterboxResult
        {
            std::vector<float> data{};
            float              scale{1.0F};
            float              pad_x{0.0F};
            float              pad_y{0.0F};
            int                input_width{0};
            int                input_height{0};
        };

        struct Candidate
        {
            cvkit::core::BBox box{};
            float             score{0.0F};
            int               class_id{-1};
        };

        Ort::Env& ort_env()
        {
            static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "cvkit"};
            return env;
        }

        [[nodiscard]] bool is_nchw_layout(const std::vector<std::int64_t>& shape)
        {
            return shape.size() == 4 && shape[1] > 0 && shape[1] <= 4;
        }

        [[nodiscard]] std::vector<std::int64_t> resolve_input_shape(
            const std::vector<std::int64_t>& input_shape,
            const cvkit::core::Frame&        frame)
        {
            auto resolved = input_shape;
            if (resolved.empty())
            {
                return resolved;
            }

            const auto channels = std::max(1, frame.desc.channels);
            const auto height   = std::max(1, frame.desc.height);
            const auto width    = std::max(1, frame.desc.width);

            if (resolved.size() == 4)
            {
                if (resolved[0] <= 0)
                {
                    resolved[0] = 1;
                }

                if (is_nchw_layout(resolved))
                {
                    if (resolved[1] <= 0)
                    {
                        resolved[1] = channels;
                    }
                    if (resolved[2] <= 0)
                    {
                        resolved[2] = height;
                    }
                    if (resolved[3] <= 0)
                    {
                        resolved[3] = width;
                    }
                }
                else
                {
                    if (resolved[1] <= 0)
                    {
                        resolved[1] = height;
                    }
                    if (resolved[2] <= 0)
                    {
                        resolved[2] = width;
                    }
                    if (resolved[3] <= 0)
                    {
                        resolved[3] = channels;
                    }
                }
            }

            return resolved;
        }

        [[nodiscard]] cv::Mat frame_to_mat(const cvkit::core::Frame& frame)
        {
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0 || frame.data.empty())
            {
                return {};
            }

            const auto type = frame.desc.channels == 3 ? CV_8UC3 : CV_8UC1;
            cv::Mat    mat(frame.desc.height, frame.desc.width, type, const_cast<std::uint8_t*>(frame.data.data()));
            return mat.clone();
        }

        [[nodiscard]] LetterboxResult preprocess_yolo(
            const cvkit::core::Frame&        frame,
            const std::vector<std::int64_t>& input_shape)
        {
            LetterboxResult result{};
            if (!is_nchw_layout(input_shape))
            {
                return result;
            }

            const auto input_h = static_cast<int>(input_shape[2]);
            const auto input_w = static_cast<int>(input_shape[3]);
            if (input_h <= 0 || input_w <= 0)
            {
                return result;
            }

            auto source = frame_to_mat(frame);
            if (source.empty())
            {
                return result;
            }

            if (source.channels() == 3)
            {
                if (frame.desc.format != cvkit::core::PixelFormat::rgb8)
                {
                    cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
                }
            }

            const auto src_w     = static_cast<float>(source.cols);
            const auto src_h     = static_cast<float>(source.rows);
            const auto scale     = std::min(static_cast<float>(input_w) / src_w, static_cast<float>(input_h) / src_h);
            const auto resized_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
            const auto resized_h = std::max(1, static_cast<int>(std::round(src_h * scale)));

            cv::Mat    resized;
            cv::resize(source, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat    canvas(input_h, input_w, resized.type(), cv::Scalar(114, 114, 114));
            const auto pad_x = (input_w - resized_w) / 2;
            const auto pad_y = (input_h - resized_h) / 2;
            resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

            cv::Mat input_float;
            canvas.convertTo(input_float, CV_32F, 1.0 / 255.0);

            const auto channels   = static_cast<std::size_t>(input_float.channels());
            const auto plane_size = static_cast<std::size_t>(input_w * input_h);
            result.data.assign(channels * plane_size, 0.0F);

            if (channels == 3)
            {
                std::vector<cv::Mat> split_channels;
                cv::split(input_float, split_channels);
                for (std::size_t c = 0; c < channels; ++c)
                {
                    std::memcpy(
                        result.data.data() + c * plane_size,
                        split_channels[c].ptr<float>(),
                        plane_size * sizeof(float));
                }
            }
            else
            {
                std::memcpy(result.data.data(), input_float.ptr<float>(), plane_size * sizeof(float));
            }

            result.scale        = scale;
            result.pad_x        = static_cast<float>(pad_x);
            result.pad_y        = static_cast<float>(pad_y);
            result.input_width  = input_w;
            result.input_height = input_h;
            return result;
        }

        [[nodiscard]] float intersection_over_union(const cvkit::core::BBox& lhs, const cvkit::core::BBox& rhs)
        {
            const auto left   = std::max(lhs.x, rhs.x);
            const auto top    = std::max(lhs.y, rhs.y);
            const auto right  = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
            const auto bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);

            const auto width        = std::max(0.0F, right - left);
            const auto height       = std::max(0.0F, bottom - top);
            const auto intersection = width * height;
            const auto union_area   = lhs.width * lhs.height + rhs.width * rhs.height - intersection;
            if (union_area <= 0.0F)
            {
                return 0.0F;
            }

            return intersection / union_area;
        }

        [[nodiscard]] std::string trim_ascii(std::string value)
        {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
            {
                return {};
            }

            const auto last = value.find_last_not_of(" \t\r\n");
            return value.substr(first, last - first + 1);
        }

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
            const Ort::Value&         output,
            const LetterboxResult&    preprocess,
            const cvkit::core::Frame& frame,
            float                     confidence_threshold)
        {
            std::vector<Candidate> candidates;
            if (!output.IsTensor())
            {
                return candidates;
            }

            const auto info = output.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                return candidates;
            }

            const auto shape = info.GetShape();
            const auto count = info.GetElementCount();
            if (count == 0)
            {
                return candidates;
            }

            const auto* data = output.GetTensorData<float>();
            if (data == nullptr)
            {
                return candidates;
            }

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

            if (boxes == 0 || attrs <= 4)
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

        [[nodiscard]] std::vector<cvkit::core::Detection> non_maximum_suppression(
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
                if (keep.class_id >= 0 && static_cast<std::size_t>(keep.class_id) < labels.size())
                {
                    detection.label = labels[static_cast<std::size_t>(keep.class_id)];
                }
                else
                {
                    detection.label = "class_" + std::to_string(keep.class_id);
                }
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

    }  // namespace
#endif

    class Model::Impl
    {
      public:
        bool load(std::string model_path)
        {
            model_path_ = std::move(model_path);
            session_.reset();
            loaded_  = false;
            backend_ = Backend::none;

            if (model_path_.empty() || !std::filesystem::exists(model_path_))
            {
                return false;
            }

#if defined(CVKIT_WITH_ONNXRUNTIME)
            try
            {
                Ort::SessionOptions session_options;
                session_options.SetIntraOpNumThreads(1);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                session_ = std::make_unique<Ort::Session>(ort_env(), model_path_.c_str(), session_options);
                loaded_  = true;
                backend_ = Backend::onnxruntime;
                return true;
            }
            catch (const Ort::Exception&)
            {
                session_.reset();
                loaded_  = false;
                backend_ = Backend::none;
                return false;
            }
#else
            return false;
#endif
        }

        bool load_labels(std::string labels_path)
        {
            labels_.clear();
            labels_path_ = std::move(labels_path);
            if (labels_path_.empty() || !std::filesystem::exists(labels_path_))
            {
                return false;
            }

            std::ifstream input(labels_path_);
            if (!input.is_open())
            {
                labels_path_.clear();
                return false;
            }

            for (std::string line; std::getline(input, line);)
            {
                auto trimmed = trim_ascii(std::move(line));
                if (trimmed.empty() || trimmed.front() == '#')
                {
                    continue;
                }

                labels_.push_back(std::move(trimmed));
            }

            if (labels_.empty())
            {
                labels_path_.clear();
                return false;
            }

            return true;
        }

        [[nodiscard]] bool loaded() const
        {
            return loaded_;
        }

        [[nodiscard]] Backend backend() const
        {
            return backend_;
        }

        [[nodiscard]] std::string_view model_path() const
        {
            return model_path_;
        }

        [[nodiscard]] std::string_view labels_path() const
        {
            return labels_path_;
        }

        void set_confidence_threshold(float threshold)
        {
            confidence_threshold_ = std::clamp(threshold, 0.0F, 1.0F);
        }

        [[nodiscard]] float confidence_threshold() const
        {
            return confidence_threshold_;
        }

        void set_iou_threshold(float threshold)
        {
            iou_threshold_ = std::clamp(threshold, 0.0F, 1.0F);
        }

        [[nodiscard]] float iou_threshold() const
        {
            return iou_threshold_;
        }

        [[nodiscard]] std::vector<cvkit::core::Detection> run(const cvkit::core::Frame& frame) const
        {
#if defined(CVKIT_WITH_ONNXRUNTIME)
            if (!loaded_ || session_ == nullptr)
            {
                return {};
            }

            try
            {
                Ort::AllocatorWithDefaultOptions allocator;

                const auto                       input_count = session_->GetInputCount();
                if (input_count == 0)
                {
                    return {};
                }

                const auto input_name        = session_->GetInputNameAllocated(0, allocator);
                auto       input_type_info   = session_->GetInputTypeInfo(0);
                auto       input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

                if (input_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    return {};
                }

                const auto resolved_shape = resolve_input_shape(input_tensor_info.GetShape(), frame);
                auto       preprocess     = preprocess_yolo(frame, resolved_shape);
                if (preprocess.data.empty())
                {
                    return {};
                }

                std::array<std::int64_t, 4> tensor_shape{
                    1,
                    3,
                    static_cast<std::int64_t>(preprocess.input_height),
                    static_cast<std::int64_t>(preprocess.input_width),
                };

                Ort::MemoryInfo memory_info  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                auto            input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    preprocess.data.data(),
                    preprocess.data.size(),
                    tensor_shape.data(),
                    tensor_shape.size());

                std::vector<const char*> input_names{input_name.get()};
                std::vector<Ort::Value>  input_tensors;
                input_tensors.emplace_back(std::move(input_tensor));

                std::vector<Ort::AllocatedStringPtr> output_name_storage;
                std::vector<const char*>             output_names;
                output_name_storage.reserve(session_->GetOutputCount());
                output_names.reserve(session_->GetOutputCount());
                for (std::size_t i = 0; i < session_->GetOutputCount(); ++i)
                {
                    output_name_storage.emplace_back(session_->GetOutputNameAllocated(i, allocator));
                    output_names.push_back(output_name_storage.back().get());
                }

                auto outputs = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names.data(),
                    input_tensors.data(),
                    input_tensors.size(),
                    output_names.data(),
                    output_names.size());

                std::vector<Candidate> candidates;
                for (const auto& output : outputs)
                {
                    auto parsed = parse_yolo_output_tensor(output, preprocess, frame, confidence_threshold_);
                    candidates.insert(candidates.end(), parsed.begin(), parsed.end());
                }

                return non_maximum_suppression(std::move(candidates), labels_, iou_threshold_);
            }
            catch (const Ort::Exception&)
            {
                return {};
            }
#else
            static_cast<void>(frame);
            return {};
#endif
        }

      private:
        std::string model_path_{};
        std::string labels_path_{};
#if defined(CVKIT_WITH_ONNXRUNTIME)
        std::unique_ptr<Ort::Session> session_{};
#else
        std::unique_ptr<int> session_{};
#endif
        std::vector<std::string> labels_{};
        bool                     loaded_{false};
        Backend                  backend_{Backend::none};
        float                    confidence_threshold_{0.25F};
        float                    iou_threshold_{0.45F};
    };

    Model::Model()
        : impl_(std::make_unique<Impl>())
    {
    }
    Model::~Model()                           = default;
    Model::Model(Model&&) noexcept            = default;
    Model& Model::operator=(Model&&) noexcept = default;

    bool   Model::load(std::string model_path)
    {
        return impl_->load(std::move(model_path));
    }

    bool Model::load_labels(std::string labels_path)
    {
        return impl_->load_labels(std::move(labels_path));
    }

    bool Model::loaded() const
    {
        return impl_->loaded();
    }

    Backend Model::backend() const
    {
        return impl_->backend();
    }

    std::string_view Model::model_path() const
    {
        return impl_->model_path();
    }

    std::string_view Model::labels_path() const
    {
        return impl_->labels_path();
    }

    void Model::set_confidence_threshold(float threshold)
    {
        impl_->set_confidence_threshold(threshold);
    }

    float Model::confidence_threshold() const
    {
        return impl_->confidence_threshold();
    }

    void Model::set_iou_threshold(float threshold)
    {
        impl_->set_iou_threshold(threshold);
    }

    float Model::iou_threshold() const
    {
        return impl_->iou_threshold();
    }

    std::vector<cvkit::core::Detection> Model::run(const cvkit::core::Frame& frame) const
    {
        return impl_->run(frame);
    }

}  // namespace cvkit::infer
