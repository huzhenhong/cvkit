#pragma once

#include "cvkit/infer/backend.h"
#include "cvkit/infer/model.h"
#include "cvkit/infer/session.h"

#include <chrono>
#include <future>
#include <memory>
#include <vector>

namespace cvkit::infer::detail
{

    using RawTensor    = cvkit::infer::TensorValue;
    using RawTensorMap = cvkit::infer::TensorMap;

    class BackendFuture
    {
      public:
        BackendFuture() = default;
        explicit BackendFuture(std::shared_future<RawTensorMap> future)
            : future_(std::move(future))
        {
        }

        [[nodiscard]] bool valid() const
        {
            return future_.valid();
        }

        RawTensorMap get()
        {
            return future_.get();
        }

        template<typename Rep, typename Period>
        [[nodiscard]] std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeout) const
        {
            return future_.wait_for(timeout);
        }

      private:
        std::shared_future<RawTensorMap> future_{};
    };

    [[nodiscard]] inline BackendFuture make_ready_backend_future(RawTensorMap outputs)
    {
        std::promise<RawTensorMap> promise;
        auto                       future = promise.get_future().share();
        promise.set_value(std::move(outputs));
        return BackendFuture{std::move(future)};
    }

    [[nodiscard]] inline bool is_supported_backend_input_tensor(const RawTensor& tensor)
    {
        return tensor.data_type == TensorDataType::float32 && tensor.memory_device == MemoryDevice::host && tensor.has_valid_host_layout();
    }

    [[nodiscard]] inline bool is_supported_backend_output_tensor_type(TensorDataType data_type)
    {
        return data_type == TensorDataType::float32;
    }

    class IBackendSession
    {
      public:
        virtual ~IBackendSession() = default;

        virtual bool              load(const ModelSpec& spec)              = 0;
        virtual bool              ready() const                            = 0;
        virtual Backend           backend() const                          = 0;
        virtual const TensorInfo* input_info(std::size_t index = 0) const  = 0;
        virtual const TensorInfo* output_info(std::size_t index = 0) const = 0;
        virtual RawTensorMap      run(const RawTensorMap& inputs) const    = 0;
        virtual bool              supports_async() const
        {
            return false;
        }
        virtual BackendFuture run_async(const RawTensorMap& inputs) const
        {
            return make_ready_backend_future(run(inputs));
        }
    };

    std::unique_ptr<IBackendSession> create_backend_session(Backend backend);

}  // namespace cvkit::infer::detail
