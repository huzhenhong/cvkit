#include "ort_session.h"

#if defined(CVKIT_WITH_ONNXRUNTIME)
    #include <onnxruntime_cxx_api.h>
#endif

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{
#if defined(CVKIT_WITH_ONNXRUNTIME)
    namespace
    {

        Ort::Env& ort_env()
        {
            static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "cvkit"};
            return env;
        }

    }  // namespace
#endif

    bool OrtSession::load(const ModelSpec& spec)
    {
        ready_ = false;
#if defined(CVKIT_WITH_ONNXRUNTIME)
        session_.reset();
        input_names_.clear();
        output_names_.clear();
        input_infos_.clear();
        output_infos_.clear();

        if (spec.model_path.empty() || !std::filesystem::exists(spec.model_path))
        {
            return false;
        }

        try
        {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(ort_env(), spec.model_path.c_str(), session_options);

            Ort::AllocatorWithDefaultOptions allocator;
            input_names_.reserve(session_->GetInputCount());
            output_names_.reserve(session_->GetOutputCount());

            for (std::size_t i = 0; i < session_->GetInputCount(); ++i)
            {
                auto name = session_->GetInputNameAllocated(i, allocator);
                input_names_.emplace_back(name.get());
                input_infos_.push_back(TensorInfo{
                    input_names_.back(),
                    session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
            }

            for (std::size_t i = 0; i < session_->GetOutputCount(); ++i)
            {
                auto name = session_->GetOutputNameAllocated(i, allocator);
                output_names_.emplace_back(name.get());
                output_infos_.push_back(TensorInfo{
                    output_names_.back(),
                    session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
            }

            ready_ = true;
            return true;
        }
        catch (const Ort::Exception&)
        {
            session_.reset();
            input_names_.clear();
            output_names_.clear();
            input_infos_.clear();
            output_infos_.clear();
            return false;
        }
#else
        static_cast<void>(spec);
        return false;
#endif
    }

    bool OrtSession::ready() const
    {
        return ready_;
    }

    Backend OrtSession::backend() const
    {
        return Backend::onnxruntime;
    }

    const TensorInfo* OrtSession::input_info(std::size_t index) const
    {
#if defined(CVKIT_WITH_ONNXRUNTIME)
        if (index < input_infos_.size())
        {
            return &input_infos_[index];
        }
#else
        static_cast<void>(index);
#endif
        return nullptr;
    }

    const TensorInfo* OrtSession::output_info(std::size_t index) const
    {
#if defined(CVKIT_WITH_ONNXRUNTIME)
        if (index < output_infos_.size())
        {
            return &output_infos_[index];
        }
#else
        static_cast<void>(index);
#endif
        return nullptr;
    }

    RawTensorMap OrtSession::run(const RawTensorMap& inputs) const
    {
        RawTensorMap outputs{};
#if defined(CVKIT_WITH_ONNXRUNTIME)
        if (!ready_ || session_ == nullptr || inputs.empty())
        {
            return outputs;
        }

        try
        {
            Ort::MemoryInfo       memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value> ort_inputs;
            std::vector<const char*> input_names;

            ort_inputs.reserve(inputs.size());
            input_names.reserve(inputs.size());

            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const auto& input = inputs[i];
                if (input.data.empty() || input.shape.empty())
                {
                    continue;
                }

                input_names.push_back(
                    i < input_names_.size() ? input_names_[i].c_str() : input.name.c_str());
                ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    const_cast<float*>(input.data.data()),
                    input.data.size(),
                    input.shape.data(),
                    input.shape.size()));
            }

            if (ort_inputs.empty())
            {
                return outputs;
            }

            std::vector<const char*> output_names;
            output_names.reserve(output_names_.size());
            for (const auto& name : output_names_)
            {
                output_names.push_back(name.c_str());
            }

            auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                ort_inputs.data(),
                ort_inputs.size(),
                output_names.data(),
                output_names.size());

            outputs.reserve(ort_outputs.size());
            for (std::size_t i = 0; i < ort_outputs.size(); ++i)
            {
                const auto& value = ort_outputs[i];
                if (!value.IsTensor())
                {
                    continue;
                }

                const auto info = value.GetTensorTypeAndShapeInfo();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    continue;
                }

                RawTensor tensor{};
                tensor.name  = i < output_names_.size() ? output_names_[i] : std::string{};
                tensor.shape = info.GetShape();
                const auto* data = value.GetTensorData<float>();
                const auto  size = info.GetElementCount();
                if (data != nullptr && size > 0)
                {
                    tensor.data.assign(data, data + size);
                }
                outputs.push_back(std::move(tensor));
            }
        }
        catch (const Ort::Exception&)
        {
            return {};
        }
#else
        static_cast<void>(inputs);
#endif
        return outputs;
    }

}  // namespace cvkit::infer::detail
