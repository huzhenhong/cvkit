#pragma once

#include "cvkit/infer/infer_export.h"

#include "cvkit/core/types.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cvkit::infer
{

    enum class BK_INFER_EXPORT Backend : unsigned char
    {
        none,
        onnxruntime,
        openvino,
        tensorrt,
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

        bool                                              load(std::string model_path);
        bool                                              load_labels(std::string labels_path);
        [[nodiscard]] bool                                loaded() const;
        [[nodiscard]] Backend                             backend() const;
        [[nodiscard]] std::string_view                    model_path() const;
        [[nodiscard]] std::string_view                    labels_path() const;
        void                                              set_confidence_threshold(float threshold);
        [[nodiscard]] float                               confidence_threshold() const;
        void                                              set_iou_threshold(float threshold);
        [[nodiscard]] float                               iou_threshold() const;
        [[nodiscard]] std::vector<cvkit::core::Detection> run(const cvkit::core::Frame& frame) const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::infer
