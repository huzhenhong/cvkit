#pragma once

#include "../backend_session.h"

#include <memory>

namespace cvkit::infer::detail
{

    class TrtSession final : public IBackendSession
    {
      public:
        TrtSession();
        ~TrtSession() override;

        TrtSession(TrtSession&&) noexcept;
        TrtSession& operator=(TrtSession&&) noexcept;

        TrtSession(const TrtSession&)                  = delete;
        TrtSession&       operator=(const TrtSession&) = delete;

        bool              load(const ModelSpec& spec) override;
        bool              ready() const override;
        Backend           backend() const override;
        const TensorInfo* input_info(std::size_t index = 0) const override;
        const TensorInfo* output_info(std::size_t index = 0) const override;
        RawTensorMap      run(const RawTensorMap& inputs) const override;
        bool              supports_async() const override;
        BackendFuture     run_async(const RawTensorMap& inputs) const override;

      private:
        class Impl;
        std::shared_ptr<Impl> impl_;
    };

}  // namespace cvkit::infer::detail
