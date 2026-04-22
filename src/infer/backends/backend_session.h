#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    struct RawTensor
    {
        std::string               name{};
        std::vector<std::int64_t> shape{};
        std::vector<float>        data{};
    };

    struct TensorInfo
    {
        std::string               name{};
        std::vector<std::int64_t> shape{};
    };

    using RawTensorMap = std::vector<RawTensor>;

    class IBackendSession
    {
      public:
        virtual ~IBackendSession() = default;

        virtual bool                            load(const ModelSpec& spec)              = 0;
        [[nodiscard]] virtual bool              ready() const                            = 0;
        [[nodiscard]] virtual Backend           backend() const                          = 0;
        [[nodiscard]] virtual const TensorInfo* input_info(std::size_t index = 0) const  = 0;
        [[nodiscard]] virtual const TensorInfo* output_info(std::size_t index = 0) const = 0;
        [[nodiscard]] virtual RawTensorMap      run(const RawTensorMap& inputs) const    = 0;
    };

    [[nodiscard]] std::unique_ptr<IBackendSession> create_backend_session(Backend backend);

}  // namespace cvkit::infer::detail
