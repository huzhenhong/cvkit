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

        TrtSession(const TrtSession&) = delete;
        TrtSession& operator=(const TrtSession&) = delete;

        bool                      load(const ModelSpec& spec) override;
        [[nodiscard]] bool        ready() const override;
        [[nodiscard]] Backend     backend() const override;
        [[nodiscard]] const TensorInfo* input_info(std::size_t index = 0) const override;
        [[nodiscard]] const TensorInfo* output_info(std::size_t index = 0) const override;
        [[nodiscard]] RawTensorMap run(const RawTensorMap& inputs) const override;

      private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cvkit::infer::detail
