#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/device.h"
#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/task.h"
#include "cvkit/infer/task_io.h"
#include "cvkit/infer/task_schema.h"

#include "cvkit/core/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT CachePolicy : std::uint8_t
    {
        default_policy,
        disabled,
        read_only,
        rebuild,
    };

    struct BK_INFER_EXPORT ModelSpec
    {
        std::string model_path{};
        std::string labels_path{};
        std::string cache_dir{};
        Backend     backend{Backend::none};
        TaskKind    task{TaskKind::unknown};
        CachePolicy cache_policy{CachePolicy::default_policy};
        std::string family{};
        DeviceRef   device{};
    };

    class BK_INFER_EXPORT Model
    {
      public:
        Model();
        ~Model();

        Model(Model&&) noexcept;
        Model& operator=(Model&&) noexcept;

        Model(const Model&)                                                       = delete;
        Model&                                            operator=(const Model&) = delete;

        bool                                              load(const ModelSpec& spec);
        bool                                              load(std::string model_path);
        bool                                              load_labels(std::string labels_path);
        [[nodiscard]] bool                                loaded() const;
        [[nodiscard]] Backend                             backend() const;
        [[nodiscard]] TaskKind                            task() const;
        [[nodiscard]] TaskSchema                          schema() const;
        [[nodiscard]] std::string_view                    model_path() const;
        [[nodiscard]] std::string_view                    labels_path() const;
        void                                              set_confidence_threshold(float threshold);
        [[nodiscard]] float                               confidence_threshold() const;
        void                                              set_iou_threshold(float threshold);
        [[nodiscard]] float                               iou_threshold() const;
        [[nodiscard]] TaskOutput                          run_sync(const TaskInput& input) const;
        [[nodiscard]] TaskFuture                          submit(const TaskInput& input) const;
        [[nodiscard]] std::vector<cvkit::core::Detection> run_detection(const cvkit::core::Frame& frame) const;
        [[nodiscard]] std::vector<cvkit::core::Detection> run(const cvkit::core::Frame& frame) const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::infer
