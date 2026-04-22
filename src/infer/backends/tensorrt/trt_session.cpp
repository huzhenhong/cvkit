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

        template <typename T>
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

        [[nodiscard]] bool cuda_ok(cudaError_t status)
        {
            return status == cudaSuccess;
        }

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
            std::error_code ec;
            const auto      abs_path = std::filesystem::absolute(model_path, ec).string();
            const auto      size     = std::filesystem::file_size(model_path, ec);
            const auto      mtime    = std::filesystem::last_write_time(model_path, ec).time_since_epoch().count();

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
            auto cache_dir = spec.cache_dir.empty()
                                 ? model_path.parent_path() / ".cvkit_cache" / "tensorrt"
                                 : std::filesystem::path(spec.cache_dir);
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
            return cache_dir;
        }

        [[nodiscard]] std::filesystem::path fingerprinted_engine_cache_path(
            const std::filesystem::path& model_path,
            const ModelSpec&             spec)
        {
            const auto cache_dir = cache_root(model_path, spec);

            return cache_dir
                   / (model_path.stem().string() + "."
                      + model_fingerprint(model_path) + "."
                      + runtime_fingerprint() + ".plan");
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
            nvinfer1::ICudaEngine&     engine,
            std::vector<TensorInfo>&   input_infos,
            std::vector<TensorInfo>&   output_infos)
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
                info.name  = name;
                info.shape = dims_to_shape(engine.getTensorShape(name));

                switch (engine.getTensorIOMode(name))
                {
                    case nvinfer1::TensorIOMode::kINPUT:
                        input_infos.push_back(std::move(info));
                        break;
                    case nvinfer1::TensorIOMode::kOUTPUT:
                        output_infos.push_back(std::move(info));
                        break;
                    case nvinfer1::TensorIOMode::kNONE:
                    default:
                        break;
                }
            }
        }

    }  // namespace
#endif

    class TrtSession::Impl
    {
      public:
#if defined(CVKIT_WITH_TENSORRT)
        TrtLogger logger{};
        std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>> builder{};
        std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network{};
        std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>> parser{};
        std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>> config{};
        std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>> plan{};
        std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime{};
        std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine{};
        std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context{};
#endif
        std::vector<TensorInfo> input_infos{};
        std::vector<TensorInfo> output_infos{};
        bool                    ready{false};
    };

    TrtSession::TrtSession()
        : impl_(std::make_unique<Impl>())
    {
    }

    TrtSession::~TrtSession() = default;

    TrtSession::TrtSession(TrtSession&&) noexcept            = default;
    TrtSession& TrtSession::operator=(TrtSession&&) noexcept = default;

    bool TrtSession::load(const ModelSpec& spec)
    {
        impl_->ready = false;
        impl_->input_infos.clear();
        impl_->output_infos.clear();

#if defined(CVKIT_WITH_TENSORRT)
        impl_->builder.reset();
        impl_->network.reset();
        impl_->parser.reset();
        impl_->config.reset();
        impl_->plan.reset();
        impl_->runtime.reset();
        impl_->engine.reset();
        impl_->context.reset();

        if (spec.model_path.empty() || !std::filesystem::exists(spec.model_path))
        {
            return false;
        }

        try
        {
            impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
            if (!impl_->runtime)
            {
                return false;
            }

            const std::filesystem::path model_path{spec.model_path};
            const auto                  caches = cache_candidates(model_path, spec);

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

                impl_->engine.reset(
                    impl_->runtime->deserializeCudaEngine(cache_bytes.data(), cache_bytes.size()));
                if (impl_->engine)
                {
                    if (spec.cache_policy != CachePolicy::read_only
                        && !caches.empty()
                        && cache_path != caches.front())
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
                    break;
                }
            }

            if (!impl_->engine)
            {
                impl_->builder.reset(nvinfer1::createInferBuilder(impl_->logger));
                if (!impl_->builder)
                {
                    return false;
                }

                const auto explicit_batch =
                    1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
                impl_->network.reset(impl_->builder->createNetworkV2(explicit_batch));
                if (!impl_->network)
                {
                    return false;
                }

                impl_->parser.reset(nvonnxparser::createParser(*impl_->network, impl_->logger));
                if (!impl_->parser)
                {
                    return false;
                }

                if (!impl_->parser->parseFromFile(
                        spec.model_path.c_str(),
                        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
                {
                    return false;
                }

                impl_->config.reset(impl_->builder->createBuilderConfig());
                if (!impl_->config)
                {
                    return false;
                }

                impl_->config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30U);
                impl_->plan.reset(impl_->builder->buildSerializedNetwork(*impl_->network, *impl_->config));
                if (!impl_->plan)
                {
                    return false;
                }

                if (spec.cache_policy != CachePolicy::disabled
                    && spec.cache_policy != CachePolicy::read_only)
                {
                    write_binary_file(
                        fingerprinted_engine_cache_path(model_path, spec),
                        impl_->plan->data(),
                        impl_->plan->size());
                }
                impl_->engine.reset(
                    impl_->runtime->deserializeCudaEngine(impl_->plan->data(), impl_->plan->size()));
            }

            if (!impl_->engine)
            {
                return false;
            }

            impl_->context.reset(impl_->engine->createExecutionContext());
            if (!impl_->context)
            {
                return false;
            }

            collect_engine_io(*impl_->engine, impl_->input_infos, impl_->output_infos);

            impl_->ready = true;
            return true;
        }
        catch (...)
        {
            impl_->builder.reset();
            impl_->network.reset();
            impl_->parser.reset();
            impl_->config.reset();
            impl_->plan.reset();
            impl_->runtime.reset();
            impl_->engine.reset();
            impl_->context.reset();
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
        RawTensorMap outputs{};
        if (!impl_->ready || !impl_->engine || !impl_->context || inputs.empty())
        {
            return outputs;
        }

        cudaStream_t stream{};
        if (!cuda_ok(cudaStreamCreate(&stream)))
        {
            return outputs;
        }

        struct StreamGuard
        {
            cudaStream_t stream{};
            ~StreamGuard()
            {
                if (stream != nullptr)
                {
                    cudaStreamDestroy(stream);
                }
            }
        } stream_guard{stream};

        std::vector<void*> device_buffers;
        auto cleanup = [&device_buffers]()
        {
            for (auto* ptr : device_buffers)
            {
                if (ptr != nullptr)
                {
                    cudaFree(ptr);
                }
            }
        };

        const auto io_count = impl_->engine->getNbIOTensors();
        device_buffers.reserve(static_cast<std::size_t>(io_count));

        std::size_t input_index = 0;
        for (int i = 0; i < io_count; ++i)
        {
            const auto* name = impl_->engine->getIOTensorName(i);
            if (name == nullptr)
            {
                cleanup();
                return {};
            }

            const auto mode = impl_->engine->getTensorIOMode(name);
            const auto type = impl_->engine->getTensorDataType(name);
            if (!is_float_tensor(type))
            {
                cleanup();
                return {};
            }

            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                if (input_index >= inputs.size())
                {
                    cleanup();
                    return {};
                }

                const auto& input = inputs[input_index++];
                if (input.shape.empty() || input.data.empty())
                {
                    cleanup();
                    return {};
                }

                if (!impl_->context->setInputShape(name, shape_to_dims(input.shape)))
                {
                    cleanup();
                    return {};
                }

                const auto bytes = input.data.size() * sizeof(float);
                void*      device_ptr{};
                if (!cuda_ok(cudaMalloc(&device_ptr, bytes)))
                {
                    cleanup();
                    return {};
                }
                device_buffers.push_back(device_ptr);

                if (!cuda_ok(cudaMemcpyAsync(device_ptr, input.data.data(), bytes, cudaMemcpyHostToDevice, stream)))
                {
                    cleanup();
                    return {};
                }

                if (!impl_->context->setTensorAddress(name, device_ptr))
                {
                    cleanup();
                    return {};
                }
            }
            else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                const auto shape = dims_to_shape(impl_->context->getTensorShape(name));
                const auto count = element_count(shape);
                if (count == 0)
                {
                    cleanup();
                    return {};
                }

                void* device_ptr{};
                if (!cuda_ok(cudaMalloc(&device_ptr, count * sizeof(float))))
                {
                    cleanup();
                    return {};
                }
                device_buffers.push_back(device_ptr);

                if (!impl_->context->setTensorAddress(name, device_ptr))
                {
                    cleanup();
                    return {};
                }
            }
        }

        if (!impl_->context->enqueueV3(stream))
        {
            cleanup();
            return {};
        }

        outputs.reserve(impl_->output_infos.size());
        for (const auto& output_info : impl_->output_infos)
        {
            const auto shape = dims_to_shape(impl_->context->getTensorShape(output_info.name.c_str()));
            const auto count = element_count(shape);
            if (count == 0)
            {
                cleanup();
                return {};
            }

            const auto* device_ptr = impl_->context->getTensorAddress(output_info.name.c_str());
            if (device_ptr == nullptr)
            {
                cleanup();
                return {};
            }

            RawTensor output{};
            output.name  = output_info.name;
            output.shape = shape;
            output.data.resize(count);

            if (!cuda_ok(cudaMemcpyAsync(
                    output.data.data(),
                    device_ptr,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream)))
            {
                cleanup();
                return {};
            }

            outputs.push_back(std::move(output));
        }

        if (!cuda_ok(cudaStreamSynchronize(stream)))
        {
            cleanup();
            return {};
        }

        cleanup();
        return outputs;
#else
        static_cast<void>(inputs);
        return {};
#endif
    }

}  // namespace cvkit::infer::detail
