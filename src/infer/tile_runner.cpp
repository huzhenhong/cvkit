#include "tile_runner.h"

#include "utils/nms.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{
    namespace
    {

        struct TileWindow
        {
            cvkit::core::Frame frame{};
            int                x{0};
            int                y{0};
        };

        bool tile_options_enabled(const TileOptions& options, const cvkit::core::Frame& frame)
        {
            return options.enabled &&
                   options.tile_width > 0 &&
                   options.tile_height > 0 &&
                   frame.desc.width > 0 &&
                   frame.desc.height > 0 &&
                   (!frame.data.empty()) &&
                   (frame.desc.width > options.tile_width || frame.desc.height > options.tile_height);
        }

        bool supports_tiled_task(TaskKind task)
        {
            switch (task)
            {
                case TaskKind::detection:
                case TaskKind::face_detection:
                case TaskKind::classification:
                case TaskKind::segmentation:
                case TaskKind::pose:
                case TaskKind::facemesh:
                    return true;
                case TaskKind::promptable_segmentation:
                case TaskKind::unknown:
                default:
                    return false;
            }
        }

        std::vector<int> tile_axis_starts(int length, int tile_size, int overlap)
        {
            std::vector<int> starts;
            if (length <= 0 || tile_size <= 0)
            {
                return starts;
            }

            const auto clamped_tile = std::min(tile_size, length);
            const auto clamped_overlap = std::clamp(overlap, 0, clamped_tile - 1);
            const auto step = std::max(1, clamped_tile - clamped_overlap);
            const auto tail = std::max(0, length - clamped_tile);

            starts.push_back(0);
            for (int start = step; start < tail; start += step)
            {
                starts.push_back(start);
            }
            if (starts.back() != tail)
            {
                starts.push_back(tail);
            }
            return starts;
        }

        cvkit::core::Frame crop_frame(
            const cvkit::core::Frame& frame,
            int                       x,
            int                       y,
            int                       width,
            int                       height)
        {
            cvkit::core::Frame tile{};
            if (frame.desc.channels <= 0 || x < 0 || y < 0 || width <= 0 || height <= 0)
            {
                return tile;
            }

            const auto channels = static_cast<std::size_t>(frame.desc.channels);
            const auto source_width = static_cast<std::size_t>(frame.desc.width);
            const auto tile_width = static_cast<std::size_t>(width);
            const auto tile_height = static_cast<std::size_t>(height);
            const auto src_x = static_cast<std::size_t>(x);
            const auto src_y = static_cast<std::size_t>(y);
            const auto source_row_values = source_width * channels;
            const auto tile_row_values = tile_width * channels;
            if (frame.data.size() < static_cast<std::size_t>(frame.desc.height) * source_row_values)
            {
                return tile;
            }

            tile.desc.width = width;
            tile.desc.height = height;
            tile.desc.channels = frame.desc.channels;
            tile.desc.format = frame.desc.format;
            tile.pts = frame.pts;
            tile.source = frame.source;
            tile.data.resize(tile_row_values * tile_height);

            for (std::size_t row = 0; row < tile_height; ++row)
            {
                const auto src_offset = ((src_y + row) * source_row_values) + (src_x * channels);
                const auto dst_offset = row * tile_row_values;
                std::copy_n(
                    frame.data.begin() + static_cast<std::ptrdiff_t>(src_offset),
                    tile_row_values,
                    tile.data.begin() + static_cast<std::ptrdiff_t>(dst_offset));
            }
            return tile;
        }

        std::vector<TileWindow> build_tile_windows(
            const cvkit::core::Frame& frame,
            const TileOptions&        options)
        {
            std::vector<TileWindow> windows;
            if (!tile_options_enabled(options, frame))
            {
                return windows;
            }

            const auto tile_width = std::min(options.tile_width, frame.desc.width);
            const auto tile_height = std::min(options.tile_height, frame.desc.height);
            const auto x_starts = tile_axis_starts(frame.desc.width, tile_width, options.overlap_x);
            const auto y_starts = tile_axis_starts(frame.desc.height, tile_height, options.overlap_y);
            windows.reserve(x_starts.size() * y_starts.size());
            for (const auto y : y_starts)
            {
                for (const auto x : x_starts)
                {
                    const auto width = std::min(tile_width, frame.desc.width - x);
                    const auto height = std::min(tile_height, frame.desc.height - y);
                    auto tile = crop_frame(frame, x, y, width, height);
                    if (!tile.data.empty())
                    {
                        windows.push_back(TileWindow{std::move(tile), x, y});
                    }
                }
            }
            return windows;
        }

        TaskInput make_tile_input(const TaskInput& input, const cvkit::core::Frame& tile)
        {
            TaskInput tile_input = input;
            for (auto& item : tile_input.items)
            {
                if (item.name != "image")
                {
                    continue;
                }
                if (std::holds_alternative<cvkit::core::Frame>(item.value))
                {
                    item.value = tile;
                    return tile_input;
                }
                if (std::holds_alternative<ImageValue>(item.value))
                {
                    ImageValue image{};
                    image.frame = tile;
                    image.memory_device = MemoryDevice::host;
                    image.storage = StorageKind::owned;
                    item.value = std::move(image);
                    return tile_input;
                }
            }

            tile_input.add("image", tile);
            return tile_input;
        }

        void translate_detections(std::vector<cvkit::core::Detection>& detections, int offset_x, int offset_y)
        {
            for (auto& detection : detections)
            {
                detection.box.x += static_cast<float>(offset_x);
                detection.box.y += static_cast<float>(offset_y);
                for (auto& point : detection.keypoints)
                {
                    point.x += static_cast<float>(offset_x);
                    point.y += static_cast<float>(offset_y);
                }
            }
        }

        std::vector<cvkit::core::Detection> merge_detections(
            std::vector<cvkit::core::Detection> detections,
            const std::vector<std::string>&     labels,
            float                               iou_threshold)
        {
            std::vector<Candidate> candidates;
            candidates.reserve(detections.size());
            for (auto& detection : detections)
            {
                Candidate candidate{};
                candidate.box = detection.box;
                candidate.score = detection.score;
                candidate.class_id = detection.class_id;
                candidate.keypoints = std::move(detection.keypoints);
                candidates.push_back(std::move(candidate));
            }
            return non_maximum_suppression(std::move(candidates), labels, iou_threshold);
        }

        void paste_mask_tile(
            cvkit::core::Frame&       canvas,
            const cvkit::core::Frame& tile,
            int                       offset_x,
            int                       offset_y)
        {
            if (canvas.desc.channels != 1 || tile.desc.channels != 1 || tile.data.empty())
            {
                return;
            }
            for (int y = 0; y < tile.desc.height; ++y)
            {
                const auto dst_y = offset_y + y;
                if (dst_y < 0 || dst_y >= canvas.desc.height)
                {
                    continue;
                }
                for (int x = 0; x < tile.desc.width; ++x)
                {
                    const auto dst_x = offset_x + x;
                    if (dst_x < 0 || dst_x >= canvas.desc.width)
                    {
                        continue;
                    }
                    const auto src_index = static_cast<std::size_t>(y * tile.desc.width + x);
                    const auto dst_index = static_cast<std::size_t>(dst_y * canvas.desc.width + dst_x);
                    if (src_index < tile.data.size() && dst_index < canvas.data.size())
                    {
                        canvas.data[dst_index] = tile.data[src_index];
                    }
                }
            }
        }

        void translate_points(KeypointsValue& keypoints, int offset_x, int offset_y)
        {
            for (auto& point : keypoints.points)
            {
                point.x += static_cast<float>(offset_x);
                point.y += static_cast<float>(offset_y);
            }
        }

        TaskOutput merge_tile_outputs(
            const std::vector<std::pair<TileWindow, TaskOutput>>& tile_outputs,
            const cvkit::core::Frame&                             source_frame,
            TaskKind                                               task,
            const std::vector<std::string>&                       labels,
            float                                                 iou_threshold)
        {
            TaskOutput output{};
            if (tile_outputs.empty())
            {
                return output;
            }

            if (task == TaskKind::detection || task == TaskKind::face_detection)
            {
                std::vector<cvkit::core::Detection> merged_detections;
                for (const auto& [window, tile_output] : tile_outputs)
                {
                    if (const auto* detections = tile_output.find<std::vector<cvkit::core::Detection>>("detections");
                        detections != nullptr)
                    {
                        auto translated = *detections;
                        translate_detections(translated, window.x, window.y);
                        merged_detections.insert(
                            merged_detections.end(),
                            std::make_move_iterator(translated.begin()),
                            std::make_move_iterator(translated.end()));
                    }
                }
                if (!merged_detections.empty())
                {
                    output.add("detections", merge_detections(std::move(merged_detections), labels, iou_threshold));
                }
            }

            if (task == TaskKind::segmentation)
            {
                cvkit::core::Frame merged_mask{};
                bool               has_mask = false;
                for (const auto& [window, tile_output] : tile_outputs)
                {
                    const auto* mask = tile_output.find<MaskValue>("mask");
                    if (mask == nullptr || mask->frame.data.empty())
                    {
                        continue;
                    }
                    if (!has_mask)
                    {
                        merged_mask.desc.width = source_frame.desc.width;
                        merged_mask.desc.height = source_frame.desc.height;
                        merged_mask.desc.channels = 1;
                        merged_mask.desc.format = mask->frame.desc.format;
                        merged_mask.source = source_frame.source;
                        merged_mask.pts = source_frame.pts;
                        merged_mask.data.assign(
                            static_cast<std::size_t>(source_frame.desc.width * source_frame.desc.height),
                            std::uint8_t{0});
                        has_mask = true;
                    }
                    paste_mask_tile(merged_mask, mask->frame, window.x, window.y);
                }
                if (has_mask)
                {
                    output.add("mask", MaskValue{std::move(merged_mask)});
                }
            }

            if (task == TaskKind::classification)
            {
                std::vector<float> score_sum;
                std::size_t        score_count = 0;
                for (const auto& [window, tile_output] : tile_outputs)
                {
                    static_cast<void>(window);
                    if (const auto* scores = tile_output.find<std::vector<float>>("scores");
                        scores != nullptr && !scores->empty())
                    {
                        if (score_sum.size() < scores->size())
                        {
                            score_sum.resize(scores->size(), 0.0F);
                        }
                        for (std::size_t index = 0; index < scores->size(); ++index)
                        {
                            score_sum[index] += (*scores)[index];
                        }
                        ++score_count;
                    }
                }
                if (score_count > 0U)
                {
                    for (auto& score : score_sum)
                    {
                        score /= static_cast<float>(score_count);
                    }
                    const auto best = std::max_element(score_sum.begin(), score_sum.end());
                    if (best != score_sum.end())
                    {
                        ClassificationValue classification{};
                        classification.class_id = static_cast<int>(std::distance(score_sum.begin(), best));
                        classification.score = *best;
                        if (classification.class_id >= 0 &&
                            static_cast<std::size_t>(classification.class_id) < labels.size())
                        {
                            classification.label = labels[static_cast<std::size_t>(classification.class_id)];
                        }
                        output.add("classification", classification);
                    }
                    output.add("scores", std::move(score_sum));
                }
            }

            if (task == TaskKind::pose)
            {
                KeypointsValue merged_keypoints{};
                for (const auto& [window, tile_output] : tile_outputs)
                {
                    if (const auto* keypoints = tile_output.find<KeypointsValue>("keypoints"); keypoints != nullptr)
                    {
                        auto translated = *keypoints;
                        translate_points(translated, window.x, window.y);
                        merged_keypoints.points.insert(
                            merged_keypoints.points.end(),
                            translated.points.begin(),
                            translated.points.end());
                    }
                }
                if (!merged_keypoints.points.empty())
                {
                    output.add("keypoints", std::move(merged_keypoints));
                }
            }

            if (task == TaskKind::facemesh)
            {
                KeypointsValue merged_landmarks{};
                for (const auto& [window, tile_output] : tile_outputs)
                {
                    if (const auto* landmarks = tile_output.find<KeypointsValue>("landmarks"); landmarks != nullptr)
                    {
                        auto translated = *landmarks;
                        translate_points(translated, window.x, window.y);
                        merged_landmarks.points.insert(
                            merged_landmarks.points.end(),
                            translated.points.begin(),
                            translated.points.end());
                    }
                }
                if (!merged_landmarks.points.empty())
                {
                    output.add("landmarks", std::move(merged_landmarks));
                }
            }

            return output;
        }

    }  // namespace

    const cvkit::core::Frame* find_tiled_source_frame(const TaskInput& input)
    {
        if (const auto* image = input.find<ImageValue>("image"); image != nullptr)
        {
            return image->has_valid_host_layout() ? &image->frame : nullptr;
        }
        return input.find<cvkit::core::Frame>("image");
    }

    bool should_run_tiled(TaskKind task, const TileOptions& options, const cvkit::core::Frame* frame)
    {
        return supports_tiled_task(task) && frame != nullptr && tile_options_enabled(options, *frame);
    }

    TaskOutput run_tiled_sync(
        const std::shared_ptr<IBackendSession>& backend,
        const std::shared_ptr<ITaskPipeline>&   pipeline,
        const PipelineContext&                  context,
        const TaskInput&                        input,
        const TileOptions&                      options,
        const cvkit::core::Frame&               source_frame)
    {
        const auto windows = build_tile_windows(source_frame, options);
        if (windows.empty())
        {
            return {};
        }

        std::vector<std::pair<TileWindow, TaskOutput>> tile_outputs;
        tile_outputs.reserve(windows.size());
        for (auto window : windows)
        {
            auto tile_input = make_tile_input(input, window.frame);
            auto tile_output = pipeline->run_sync(*backend, tile_input, context);
            tile_outputs.push_back(std::make_pair(std::move(window), std::move(tile_output)));
        }
        return merge_tile_outputs(tile_outputs, source_frame, context.spec.task, context.labels, context.iou_threshold);
    }

}  // namespace cvkit::infer::detail
