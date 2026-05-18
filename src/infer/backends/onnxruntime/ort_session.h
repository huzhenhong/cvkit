#pragma once

#include "../backend_session.h"

#if defined(CVKIT_WITH_ONNXRUNTIME)
    #include <onnxruntime_cxx_api.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    class OrtSession final : public IBackendSession
    {
      public:
        bool                      load(const ModelSpec& spec) override;
        [[nodiscard]] bool        ready() const override;
        [[nodiscard]] Backend     backend() const override;
        [[nodiscard]] const TensorInfo* input_info(std::size_t index = 0) const override;
        [[nodiscard]] const TensorInfo* output_info(std::size_t index = 0) const override;
        [[nodiscard]] RawTensorMap run(const RawTensorMap& inputs) const override;

      private:
#if defined(CVKIT_WITH_ONNXRUNTIME)
        std::unique_ptr<Ort::Session> session_{};
        std::vector<std::string>      input_names_{};
        std::vector<std::string>      output_names_{};
        std::vector<TensorInfo>       input_infos_{};
        std::vector<TensorInfo>       output_infos_{};
        bool                          cuda_execution_enabled_{false};
        int                           cuda_device_index_{0};
#endif
        bool ready_{false};
    };

}  // namespace cvkit::infer::detail
