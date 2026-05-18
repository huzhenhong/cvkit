#include "trt_session.h"

#if defined(CVKIT_WITH_TENSORRT)
    #include <NvInfer.h>
    #include <NvOnnxParser.h>
    #include <cuda_runtime_api.h>
#endif

#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{
#if defined(CVKIT_WITH_TENSORRT)
    namespace
    {

        template<typename T>
        struct TrtDestroy
        {
            void operator()(T* ptr) const
            {
                delete ptr;
            }
        };

        class TrtLogger final : public nvinfer1::ILogger
        {
          public:
            void log(Severity severity, const char* message) noexcept override
            {
                if (severity > Severity::kWARNING || message == nullptr)
                {
                    return;
                }
            }
        };

        [[nodiscard]] std::vector<std::int64_t> dims_to_shape(const nvinfer1::Dims& dims)
        {
            std::vector<std::int64_t> shape;
            shape.reserve(static_cast<std::size_t>(dims.nbDims));
            for (int i = 0; i < dims.nbDims; ++i)
            {
                shape.push_back(static_cast<std::int64_t>(dims.d[i]));
            }
            return shape;
        }

        [[nodiscard]] std::string to_lower(std::string_view value)
        {
            std::string result(value);
            std::transform(
                result.begin(),
                result.end(),
                result.begin(),
                [](unsigned char ch)
                {
                    return static_cast<char>(std::tolower(ch));
                });
            return result;
        }

        [[nodiscard]] const TensorRtProfileSpec* find_profile_spec(
            std::string_view                        input_name,
            const std::vector<TensorRtProfileSpec>& profiles)
        {
            const auto normalized = to_lower(input_name);
            for (const auto& profile : profiles)
            {
                if (to_lower(profile.input_name) == normalized)
                {
                    return &profile;
                }
            }
            return nullptr;
        }

        [[nodiscard]] nvinfer1::Dims shape_to_dims(const std::vector<std::int64_t>& shape)
        {
            nvinfer1::Dims dims{};
            dims.nbDims = static_cast<int>(std::min<std::size_t>(shape.size(), static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)));
            for (int i = 0; i < dims.nbDims; ++i)
            {
                dims.d[i] = static_cast<int32_t>(shape[static_cast<std::size_t>(i)]);
            }
            return dims;
        }

        [[nodiscard]] std::size_t element_count(const std::vector<std::int64_t>& shape)
        {
            if (shape.empty())
            {
                return 0;
            }

            std::size_t count = 1;
            for (const auto dim : shape)
            {
                if (dim <= 0)
                {
                    return 0;
                }
                count *= static_cast<std::size_t>(dim);
            }
            return count;
        }

        [[nodiscard]] bool is_float_tensor(nvinfer1::DataType type)
        {
            return type == nvinfer1::DataType::kFLOAT;
        }

        [[nodiscard]] bool is_supported_trt_input_tensor(const RawTensor& tensor)
        {
            if (tensor.data_type != TensorDataType::float32)
            {
                return false;
            }

            if (tensor.memory_device == MemoryDevice::host)
            {
                return tensor.has_valid_host_layout();
            }

            if (tensor.memory_device == MemoryDevice::cuda)
            {
                return tensor.has_valid_device_view();
            }

            return false;
        }

        [[nodiscard]] TensorDataType trt_data_type(nvinfer1::DataType type)
        {
            switch (type)
            {
                case nvinfer1::DataType::kFLOAT:
                    return TensorDataType::float32;
                case nvinfer1::DataType::kHALF:
                    return TensorDataType::float16;
                case nvinfer1::DataType::kINT32:
                    return TensorDataType::int32;
    #if NV_TENSORRT_MAJOR >= 10
                case nvinfer1::DataType::kINT64:
                    return TensorDataType::int64;
                case nvinfer1::DataType::kUINT8:
                    return TensorDataType::uint8;
                case nvinfer1::DataType::kBOOL:
                    return TensorDataType::boolean;
    #endif
                default:
                    return TensorDataType::unknown;
            }
        }

        [[nodiscard]] bool has_dynamic_dims(const nvinfer1::Dims& dims)
        {
            for (int i = 0; i < dims.nbDims; ++i)
            {
                if (dims.d[i] < 0)
                {
                    return true;
                }
            }
            return false;
        }

        enum class ProfilePreset : std::uint8_t
        {
            min,
            opt,
            max,
        };

        [[nodiscard]] int32_t resolve_dynamic_dim(
            std::string_view input_name,
            const ModelSpec& spec,
            int              nb_dims,
            int              dim_index,
            ProfilePreset    preset)
        {
            const auto name          = to_lower(input_name);
            const auto spatial_value = [preset](int min_value, int opt_value, int max_value)
            {
                switch (preset)
                {
                    case ProfilePreset::min:
                        return min_value;
                    case ProfilePreset::max:
                        return max_value;
                    case ProfilePreset::opt:
                    default:
                        return opt_value;
                }
            };

            if (dim_index == 0)
            {
                return 1;
            }

            if (spec.task == TaskKind::detection && nb_dims == 4)
            {
                if (dim_index == 1)
                {
                    return 3;
                }
                if (dim_index == 2 || dim_index == 3)
                {
                    return spatial_value(320, 640, 1280);
                }
            }

            if (spec.task == TaskKind::promptable_segmentation)
            {
                if (name.find("embedding") != std::string::npos && nb_dims == 4)
                {
                    if (dim_index == 1)
                    {
                        return 256;
                    }
                    if (dim_index == 2 || dim_index == 3)
                    {
                        return 64;
                    }
                }

                if ((name.find("image") != std::string::npos || name.find("images") != std::string::npos) && nb_dims == 4)
                {
                    if (dim_index == 1)
                    {
                        return 3;
                    }
                    if (dim_index == 2 || dim_index == 3)
                    {
                        return 1024;
                    }
                }

                if ((name.find("point") != std::string::npos || name.find("label") != std::string::npos) && nb_dims >= 2)
                {
                    if (dim_index == 1)
                    {
                        return spatial_value(1, 1, 32);
                    }
                    if (name.find("point") != std::string::npos && dim_index == nb_dims - 1)
                    {
                        return 2;
                    }
                    return 1;
                }

                if ((name.find("box") != std::string::npos || name.find("bbox") != std::string::npos) && nb_dims >= 2)
                {
                    if (dim_index == 1)
                    {
                        return spatial_value(1, 1, 8);
                    }
                    if (dim_index == nb_dims - 1)
                    {
                        return 4;
                    }
                    return 1;
                }
            }

            return 1;
        }

        [[nodiscard]] nvinfer1::Dims profile_dims_for_input(
            std::string_view      input_name,
            const nvinfer1::Dims& dims,
            const ModelSpec&      spec,
            ProfilePreset         preset)
        {
            if (const auto* profile = find_profile_spec(input_name, spec.tensorrt_profiles); profile != nullptr)
            {
                const auto* selected_shape = &profile->shape.opt;
                switch (preset)
                {
                    case ProfilePreset::min:
                        selected_shape = &profile->shape.min;
                        break;
                    case ProfilePreset::max:
                        selected_shape = &profile->shape.max;
                        break;
                    case ProfilePreset::opt:
                    default:
                        selected_shape = &profile->shape.opt;
                        break;
                }

                if (selected_shape->size() == static_cast<std::size_t>(dims.nbDims))
                {
                    return shape_to_dims(*selected_shape);
                }
            }

            auto resolved = dims;
            for (int i = 0; i < resolved.nbDims; ++i)
            {
                if (resolved.d[i] < 0)
                {
                    resolved.d[i] = resolve_dynamic_dim(input_name, spec, resolved.nbDims, i, preset);
                }
            }
            return resolved;
        }

        [[nodiscard]] bool cuda_ok(cudaError_t status)
        {
            return status == cudaSuccess;
        }

        struct StreamGuard
        {
            cudaStream_t       stream{};

            [[nodiscard]] bool create()
            {
                return cuda_ok(cudaStreamCreate(&stream));
            }

            ~StreamGuard()
            {
                if (stream != nullptr)
                {
                    cudaStreamDestroy(stream);
                }
            }
        };

        class DeviceBuffers
        {
          public:
            ~DeviceBuffers()
            {
                for (auto* ptr : buffers_)
                {
                    if (ptr != nullptr)
                    {
                        cudaFree(ptr);
                    }
                }
            }

            [[nodiscard]] void* allocate(std::size_t bytes)
            {
                void* ptr{};
                if (!cuda_ok(cudaMalloc(&ptr, bytes)))
                {
                    return nullptr;
                }
                buffers_.push_back(ptr);
                return ptr;
            }

            void reserve(std::size_t count)
            {
                buffers_.reserve(count);
            }

          private:
            std::vector<void*> buffers_{};
        };

        [[nodiscard]] std::filesystem::path engine_cache_path(const std::filesystem::path& model_path)
        {
            return model_path.parent_path() / (model_path.stem().string() + ".trt.plan");
        }

        [[nodiscard]] std::string to_hex(std::size_t value)
        {
            std::ostringstream stream;
            stream << std::hex << value;
            return stream.str();
        }

        [[nodiscard]] std::string model_fingerprint(const std::filesystem::path& model_path)
        {
            std::error_code    ec;
            const auto         abs_path = std::filesystem::absolute(model_path, ec).string();
            const auto         size     = std::filesystem::file_size(model_path, ec);
            const auto         mtime    = std::filesystem::last_write_time(model_path, ec).time_since_epoch().count();

            std::ostringstream material;
            material << abs_path << '|' << size << '|' << mtime;
            return to_hex(std::hash<std::string>{}(material.str()));
        }

        [[nodiscard]] std::string runtime_fingerprint()
        {
            int device = 0;
            if (!cuda_ok(cudaGetDevice(&device)))
            {
                return "unknown";
            }

            cudaDeviceProp prop{};
            if (!cuda_ok(cudaGetDeviceProperties(&prop, device)))
            {
                return "unknown";
            }

            std::ostringstream material;
            material
                << "trt" << NV_TENSORRT_MAJOR << '_' << NV_TENSORRT_MINOR
                << ".cuda" << CUDART_VERSION
                << ".sm" << prop.major << prop.minor
                << ".gpu" << prop.name;

            return to_hex(std::hash<std::string>{}(material.str()));
        }

        [[nodiscard]] std::filesystem::path cache_root(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec)
        {
            auto            cache_dir = spec.cache_dir.empty() ? model_path.parent_path() / ".cvkit_cache" / "tensorrt" : std::filesystem::path(spec.cache_dir);
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
            return cache_dir;
        }

        [[nodiscard]] std::filesystem::path fingerprinted_engine_cache_path(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec)
        {
            const auto cache_dir = cache_root(model_path, spec);

            return cache_dir / (model_path.stem().string() + "." + model_fingerprint(model_path) + "." + runtime_fingerprint() + ".plan");
        }

        [[nodiscard]] bool is_cache_fresh(
            const std::filesystem::path& model_path,
            const std::filesystem::path& cache_path)
        {
            std::error_code ec;
            if (!std::filesystem::exists(cache_path, ec))
            {
                return false;
            }

            const auto model_time = std::filesystem::last_write_time(model_path, ec);
            if (ec)
            {
                return false;
            }

            const auto cache_time = std::filesystem::last_write_time(cache_path, ec);
            if (ec)
            {
                return false;
            }

            return cache_time >= model_time;
        }

        [[nodiscard]] std::vector<char> read_binary_file(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                return {};
            }

            const auto size = input.tellg();
            if (size <= 0)
            {
                return {};
            }

            std::vector<char> bytes(static_cast<std::size_t>(size));
            input.seekg(0, std::ios::beg);
            if (!input.read(bytes.data(), size))
            {
                return {};
            }

            return bytes;
        }

        void write_binary_file(const std::filesystem::path& path, const void* data, std::size_t size)
        {
            if (data == nullptr || size == 0)
            {
                return;
            }

            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output)
            {
                return;
            }

            output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        }

        [[nodiscard]] std::vector<std::filesystem::path> cache_candidates(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec)
        {
            std::vector<std::filesystem::path> candidates;
            if (spec.cache_policy == CachePolicy::disabled || spec.cache_policy == CachePolicy::rebuild)
            {
                return candidates;
            }

            candidates.push_back(fingerprinted_engine_cache_path(model_path, spec));
            if (spec.cache_dir.empty())
            {
                candidates.push_back(engine_cache_path(model_path));
            }
            return candidates;
        }

        void collect_engine_io(
            nvinfer1::ICudaEngine&   engine,
            bool                     prefer_device_outputs,
            std::vector<TensorInfo>& input_infos,
            std::vector<TensorInfo>& output_infos)
        {
            input_infos.clear();
            output_infos.clear();

            const auto io_count = engine.getNbIOTensors();
            for (int i = 0; i < io_count; ++i)
            {
                const auto* name = engine.getIOTensorName(i);
                if (name == nullptr)
                {
                    continue;
                }

                TensorInfo info{};
                info.name          = name;
                info.shape         = dims_to_shape(engine.getTensorShape(name));
                info.data_type     = trt_data_type(engine.getTensorDataType(name));
                info.memory_device = MemoryDevice::host;

                switch (engine.getTensorIOMode(name))
                {
                    case nvinfer1::TensorIOMode::kINPUT:
                        input_infos.push_back(std::move(info));
                        break;
                    case nvinfer1::TensorIOMode::kOUTPUT:
                        if (prefer_device_outputs)
                        {
                            info.memory_device = MemoryDevice::cuda;
                        }
                        output_infos.push_back(std::move(info));
                        break;
                    case nvinfer1::TensorIOMode::kNONE:
                    default:
                        break;
                }
            }
        }

        [[nodiscard]] bool bind_engine_tensors(
            nvinfer1::ICudaEngine&       engine,
            nvinfer1::IExecutionContext& context,
            const RawTensorMap&          inputs,
            DeviceBuffers&               device_buffers,
            cudaStream_t                 stream)
        {
            const auto  io_count = engine.getNbIOTensors();

            std::size_t input_index = 0;
            for (int i = 0; i < io_count; ++i)
            {
                const auto* name = engine.getIOTensorName(i);
                if (name == nullptr)
                {
                    return false;
                }

                const auto mode = engine.getTensorIOMode(name);
                const auto type = engine.getTensorDataType(name);
                if (!is_float_tensor(type))
                {
                    return false;
                }

                if (mode == nvinfer1::TensorIOMode::kINPUT)
                {
                    if (input_index >= inputs.size())
                    {
                        return false;
                    }

                    const auto& input = inputs[input_index++];
                    if (!is_supported_trt_input_tensor(input))
                    {
                        return false;
                    }

                    if (!context.setInputShape(name, shape_to_dims(input.shape)))
                    {
                        return false;
                    }

                    void* device_ptr = nullptr;
                    if (input.memory_device == MemoryDevice::cuda)
                    {
                        device_ptr = const_cast<void*>(input.external_data);
                    }
                    else
                    {
                        const auto bytes = input.data.size() * sizeof(float);
                        device_ptr       = device_buffers.allocate(bytes);
                        if (device_ptr == nullptr)
                        {
                            return false;
                        }

                        if (!cuda_ok(cudaMemcpyAsync(device_ptr, input.data.data(), bytes, cudaMemcpyHostToDevice, stream)))
                        {
                            return false;
                        }
                    }

                    if (!context.setTensorAddress(name, device_ptr))
                    {
                        return false;
                    }
                }
                else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
                {
                    const auto output_type = trt_data_type(type);
                    if (!is_supported_backend_output_tensor_type(output_type))
                    {
                        return false;
                    }

                    const auto shape = dims_to_shape(context.getTensorShape(name));
                    const auto count = element_count(shape);
                    if (count == 0)
                    {
                        return false;
                    }

                    auto* device_ptr = device_buffers.allocate(count * sizeof(float));
                    if (device_ptr == nullptr)
                    {
                        return false;
                    }

                    if (!context.setTensorAddress(name, device_ptr))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        [[nodiscard]] bool collect_output_tensors(
            nvinfer1::IExecutionContext&          context,
            const std::vector<TensorInfo>&        output_infos,
            const std::shared_ptr<DeviceBuffers>& device_buffers_owner,
            bool                                  prefer_device_outputs,
            cudaStream_t                          stream,
            RawTensorMap&                         outputs)
        {
            outputs.clear();
            outputs.reserve(output_infos.size());

            for (const auto& output_info : output_infos)
            {
                const auto shape = dims_to_shape(context.getTensorShape(output_info.name.c_str()));
                const auto count = element_count(shape);
                if (count == 0)
                {
                    return false;
                }

                const auto* device_ptr = context.getTensorAddress(output_info.name.c_str());
                if (device_ptr == nullptr)
                {
                    return false;
                }

                RawTensor output{};
                output.name      = output_info.name;
                output.shape     = shape;
                output.data_type = output_info.data_type;
                if (prefer_device_outputs)
                {
                    output.memory_device = MemoryDevice::cuda;
                    output.storage       = StorageKind::owned;
                    output.external_data = device_ptr;
                    output.storage_bytes = count * sizeof(float);
                    output.storage_owner = std::shared_ptr<void>(device_buffers_owner, const_cast<void*>(device_ptr));
                }
                else
                {
                    output.memory_device = MemoryDevice::host;
                    output.data.resize(count);

                    if (!cuda_ok(cudaMemcpyAsync(
                            output.data.data(),
                            device_ptr,
                            count * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream)))
                    {
                        return false;
                    }
                }

                outputs.push_back(std::move(output));
            }

            return true;
        }

    }  // namespace
#endif

    class TrtSession::Impl
    {
      public:
#if defined(CVKIT_WITH_TENSORRT)
        TrtLogger                                                                               logger{};
        std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>>                     builder{};
        std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network{};
        std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>>               parser{};
        std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>>         config{};
        std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>>               plan{};
        std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>>                     runtime{};
        std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>>               engine{};
        std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>>   context{};
        StreamGuard                                                                             execution_stream{};
        mutable std::mutex                                                                      execution_mutex{};
        ModelSpec                                                                               spec{};

        void                                                                                    reset_runtime_state();
        [[nodiscard]] bool                                                                      create_runtime();
        [[nodiscard]] bool                                                                      try_restore_engine_from_cache(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec);
        [[nodiscard]] bool build_engine_from_onnx(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec);
        [[nodiscard]] bool         configure_optimization_profiles(const ModelSpec& spec);
        [[nodiscard]] bool         create_execution_context();
        [[nodiscard]] RawTensorMap run_sync(const RawTensorMap& inputs) const;
#endif
        std::vector<TensorInfo> input_infos{};
        std::vector<TensorInfo> output_infos{};
        bool                    ready{false};
    };

#if defined(CVKIT_WITH_TENSORRT)
    void TrtSession::Impl::reset_runtime_state()
    {
        builder.reset();
        network.reset();
        parser.reset();
        config.reset();
        plan.reset();
        runtime.reset();
        engine.reset();
        context.reset();
        execution_stream = StreamGuard{};
    }

    bool TrtSession::Impl::create_runtime()
    {
        runtime.reset(nvinfer1::createInferRuntime(logger));
        return runtime != nullptr;
    }

    bool TrtSession::Impl::try_restore_engine_from_cache(
        const std::filesystem::path& model_path,
        const ModelSpec&             spec)
    {
        const auto caches = cache_candidates(model_path, spec);
        for (const auto& cache_path : caches)
        {
            if (!is_cache_fresh(model_path, cache_path))
            {
                continue;
            }

            auto cache_bytes = read_binary_file(cache_path);
            if (cache_bytes.empty())
            {
                continue;
            }

            engine.reset(runtime->deserializeCudaEngine(cache_bytes.data(), cache_bytes.size()));
            if (!engine)
            {
                continue;
            }

            if (spec.cache_policy != CachePolicy::read_only && !caches.empty() && cache_path != caches.front())
            {
                std::error_code ec;
                std::filesystem::copy_file(
                    cache_path,
                    caches.front(),
                    std::filesystem::copy_options::overwrite_existing,
                    ec);
                if (ec)
                {
                    write_binary_file(caches.front(), cache_bytes.data(), cache_bytes.size());
                }
            }
            return true;
        }

        return false;
    }

    bool TrtSession::Impl::build_engine_from_onnx(
        const std::filesystem::path& model_path,
        const ModelSpec&             spec)
    {
        builder.reset(nvinfer1::createInferBuilder(logger));
        if (!builder)
        {
            return false;
        }

        const auto explicit_batch =
            1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network.reset(builder->createNetworkV2(explicit_batch));
        if (!network)
        {
            return false;
        }

        parser.reset(nvonnxparser::createParser(*network, logger));
        if (!parser)
        {
            return false;
        }

        if (!parser->parseFromFile(
                model_path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            return false;
        }

        config.reset(builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        if (!configure_optimization_profiles(spec))
        {
            return false;
        }

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30U);
        plan.reset(builder->buildSerializedNetwork(*network, *config));
        if (!plan)
        {
            return false;
        }

        if (spec.cache_policy != CachePolicy::disabled && spec.cache_policy != CachePolicy::read_only)
        {
            write_binary_file(
                fingerprinted_engine_cache_path(model_path, spec),
                plan->data(),
                plan->size());
        }

        engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        return engine != nullptr;
    }

    bool TrtSession::Impl::configure_optimization_profiles(const ModelSpec& spec)
    {
        if (!builder || !network || !config)
        {
            return false;
        }

        bool needs_profile = false;
        for (int i = 0; i < network->getNbInputs(); ++i)
        {
            const auto* input = network->getInput(i);
            if (input != nullptr && has_dynamic_dims(input->getDimensions()))
            {
                needs_profile = true;
                break;
            }
        }

        if (!needs_profile)
        {
            return true;
        }

        auto* profile = builder->createOptimizationProfile();
        if (!profile)
        {
            return false;
        }

        for (int i = 0; i < network->getNbInputs(); ++i)
        {
            const auto* input = network->getInput(i);
            if (input == nullptr)
            {
                continue;
            }

            const auto dims = input->getDimensions();
            if (!has_dynamic_dims(dims))
            {
                continue;
            }

            const auto* name = input->getName();
            if (name == nullptr)
            {
                return false;
            }

            const auto min_dims = profile_dims_for_input(name, dims, spec, ProfilePreset::min);
            const auto opt_dims = profile_dims_for_input(name, dims, spec, ProfilePreset::opt);
            const auto max_dims = profile_dims_for_input(name, dims, spec, ProfilePreset::max);

            if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, min_dims) || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, opt_dims) || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, max_dims))
            {
                return false;
            }
        }

        if (!profile->isValid())
        {
            return false;
        }

        return config->addOptimizationProfile(profile) != -1;
    }

    bool TrtSession::Impl::create_execution_context()
    {
        if (!engine)
        {
            return false;
        }

        context.reset(engine->createExecutionContext());
        if (!context)
        {
            return false;
        }

        if (!execution_stream.create())
        {
            return false;
        }

        collect_engine_io(*engine, spec.tensorrt_prefer_device_outputs, input_infos, output_infos);
        return true;
    }

    RawTensorMap TrtSession::Impl::run_sync(const RawTensorMap& inputs) const
    {
        RawTensorMap                outputs{};
        std::lock_guard<std::mutex> lock(execution_mutex);

        if (!ready || !engine || !context || inputs.empty())
        {
            return outputs;
        }

        if (execution_stream.stream == nullptr)
        {
            return outputs;
        }

        const auto io_count       = engine->getNbIOTensors();
        auto       device_buffers = std::make_shared<DeviceBuffers>();
        device_buffers->reserve(static_cast<std::size_t>(io_count));
        if (!bind_engine_tensors(*engine, *context, inputs, *device_buffers, execution_stream.stream))
        {
            return {};
        }

        if (!context->enqueueV3(execution_stream.stream))
        {
            return {};
        }

        if (!collect_output_tensors(
                *context,
                output_infos,
                device_buffers,
                spec.tensorrt_prefer_device_outputs,
                execution_stream.stream,
                outputs))
        {
            return {};
        }

        if (!cuda_ok(cudaStreamSynchronize(execution_stream.stream)))
        {
            return {};
        }

        return outputs;
    }
#endif

    TrtSession::TrtSession()
        : impl_(std::make_shared<Impl>())
    {
    }

    TrtSession::~TrtSession() = default;

    TrtSession::TrtSession(TrtSession&&) noexcept            = default;
    TrtSession& TrtSession::operator=(TrtSession&&) noexcept = default;

    bool        TrtSession::load(const ModelSpec& spec)
    {
        impl_->ready = false;
        impl_->input_infos.clear();
        impl_->output_infos.clear();

#if defined(CVKIT_WITH_TENSORRT)
        impl_->spec = spec;
        impl_->reset_runtime_state();

        if (spec.model_path.empty() || !std::filesystem::exists(spec.model_path))
        {
            return false;
        }

        try
        {
            if (!impl_->create_runtime())
            {
                return false;
            }

            const std::filesystem::path model_path{spec.model_path};
            if (!impl_->try_restore_engine_from_cache(model_path, spec) && !impl_->build_engine_from_onnx(model_path, spec))
            {
                return false;
            }

            if (!impl_->create_execution_context())
            {
                return false;
            }

            impl_->ready = true;
            return true;
        }
        catch (...)
        {
            impl_->reset_runtime_state();
            impl_->input_infos.clear();
            impl_->output_infos.clear();
            impl_->ready = false;
            return false;
        }
#else
        static_cast<void>(spec);
        return false;
#endif
    }

    bool TrtSession::ready() const
    {
        return impl_->ready;
    }

    Backend TrtSession::backend() const
    {
        return Backend::tensorrt;
    }

    const TensorInfo* TrtSession::input_info(std::size_t index) const
    {
        if (index < impl_->input_infos.size())
        {
            return &impl_->input_infos[index];
        }
        return nullptr;
    }

    const TensorInfo* TrtSession::output_info(std::size_t index) const
    {
        if (index < impl_->output_infos.size())
        {
            return &impl_->output_infos[index];
        }
        return nullptr;
    }

    RawTensorMap TrtSession::run(const RawTensorMap& inputs) const
    {
#if defined(CVKIT_WITH_TENSORRT)
        return impl_->run_sync(inputs);
#else
        static_cast<void>(inputs);
        return {};
#endif
    }

    bool TrtSession::supports_async() const
    {
#if defined(CVKIT_WITH_TENSORRT)
        return true;
#else
        return false;
#endif
    }

    BackendFuture TrtSession::run_async(const RawTensorMap& inputs) const
    {
#if defined(CVKIT_WITH_TENSORRT)
        auto impl          = impl_;
        auto copied_inputs = inputs;
        auto future        = std::async(
                                 std::launch::async,
                                 [impl = std::move(impl), copied_inputs = std::move(copied_inputs)]() mutable
                                 {
                              return impl->run_sync(copied_inputs);
                                 })
                                 .share();
        return BackendFuture{std::move(future)};
#else
        static_cast<void>(inputs);
        return {};
#endif
    }

}  // namespace cvkit::infer::detail
