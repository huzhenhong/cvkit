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

        [[nodiscard]] TensorDataType ort_data_type(ONNXTensorElementDataType type)
        {
            switch (type)
            {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    return TensorDataType::float32;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    return TensorDataType::float16;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    return TensorDataType::int32;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    return TensorDataType::int64;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    return TensorDataType::uint8;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    return TensorDataType::boolean;
                default:
                    return TensorDataType::unknown;
            }
        }

        [[nodiscard]] bool supports_ort_cuda_input_tensor(
            const RawTensor& tensor,
            bool             cuda_execution_enabled)
        {
            return cuda_execution_enabled && tensor.data_type == TensorDataType::float32 && tensor.memory_device == MemoryDevice::cuda && tensor.has_valid_device_view();
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
        cuda_execution_enabled_ = false;
        cuda_device_index_      = 0;

        if (spec.model_path.empty() || !std::filesystem::exists(spec.model_path))
        {
            return false;
        }

        try
        {
            (void)ort_env();
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            if (spec.device.kind == DeviceKind::cuda)
            {
                OrtCUDAProviderOptions provider_options{};
                provider_options.device_id = spec.device.index;
                session_options.AppendExecutionProvider_CUDA(provider_options);
                cuda_execution_enabled_ = true;
                cuda_device_index_      = spec.device.index;
            }

            session_ = std::make_unique<Ort::Session>(ort_env(), spec.model_path.c_str(), session_options);

            Ort::AllocatorWithDefaultOptions allocator;
            input_names_.reserve(session_->GetInputCount());
            output_names_.reserve(session_->GetOutputCount());

            for (std::size_t i = 0; i < session_->GetInputCount(); ++i)
            {
                auto       name        = session_->GetInputNameAllocated(i, allocator);
                const auto tensor_info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
                input_names_.emplace_back(name.get());
                input_infos_.push_back(TensorInfo{
                    input_names_.back(),
                    tensor_info.GetShape(),
                    ort_data_type(tensor_info.GetElementType()),
                    MemoryDevice::host});
            }

            for (std::size_t i = 0; i < session_->GetOutputCount(); ++i)
            {
                auto       name        = session_->GetOutputNameAllocated(i, allocator);
                const auto tensor_info = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
                output_names_.emplace_back(name.get());
                output_infos_.push_back(TensorInfo{
                    output_names_.back(),
                    tensor_info.GetShape(),
                    ort_data_type(tensor_info.GetElementType()),
                    MemoryDevice::host});
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
            cuda_execution_enabled_ = false;
            cuda_device_index_      = 0;
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
            std::vector<Ort::Value>  ort_inputs;
            std::vector<const char*> input_names;

            ort_inputs.reserve(inputs.size());
            input_names.reserve(inputs.size());

            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const auto& input = inputs[i];
                if (!is_supported_backend_input_tensor(input) &&
                    !supports_ort_cuda_input_tensor(input, cuda_execution_enabled_))
                {
                    continue;
                }

                input_names.push_back(
                    i < input_names_.size() ? input_names_[i].c_str() : input.name.c_str());
                if (input.memory_device == MemoryDevice::host)
                {
                    auto memory_info =
                        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        const_cast<float*>(input.data.data()),
                        input.data.size(),
                        input.shape.data(),
                        input.shape.size()));
                }
                else
                {
                    auto memory_info = Ort::MemoryInfo{
                        "Cuda",
                        OrtAllocatorType::OrtDeviceAllocator,
                        cuda_device_index_,
                        OrtMemTypeDefault};
                    ort_inputs.emplace_back(Ort::Value::CreateTensor(
                        memory_info,
                        const_cast<void*>(input.external_data),
                        input.packed_byte_size(),
                        input.shape.data(),
                        input.shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
                }
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

                const auto info      = value.GetTensorTypeAndShapeInfo();
                const auto data_type = ort_data_type(info.GetElementType());
                if (!is_supported_backend_output_tensor_type(data_type))
                {
                    continue;
                }

                RawTensor tensor{};
                tensor.name          = i < output_names_.size() ? output_names_[i] : std::string{};
                tensor.shape         = info.GetShape();
                tensor.data_type     = data_type;
                tensor.memory_device = MemoryDevice::host;
                const auto* data     = value.GetTensorData<float>();
                const auto  size     = info.GetElementCount();
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
