#include "tensor_value_utils.h"

#if defined(CVKIT_WITH_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif

namespace cvkit::infer::detail
{

    std::optional<cvkit::infer::TensorValue> materialize_host_tensor(
        const cvkit::infer::TensorValue& tensor,
        std::string*                     error_message)
    {
        if (tensor.memory_device == cvkit::infer::MemoryDevice::host)
        {
            if (!tensor.has_valid_host_layout())
            {
                if (error_message != nullptr)
                {
                    *error_message = "host tensor does not expose a valid host layout";
                }
                return std::nullopt;
            }
            return tensor;
        }

        if (tensor.memory_device != cvkit::infer::MemoryDevice::cuda)
        {
            if (error_message != nullptr)
            {
                *error_message = "tensor memory device is not supported for host materialization";
            }
            return std::nullopt;
        }

        if (!tensor.has_valid_device_view())
        {
            if (error_message != nullptr)
            {
                *error_message = "cuda tensor does not expose a valid external device view";
            }
            return std::nullopt;
        }

#if !defined(CVKIT_WITH_CUDA_RUNTIME)
        if (error_message != nullptr)
        {
            *error_message = "cuda runtime is not available in this build";
        }
        return std::nullopt;
#else
        cvkit::infer::TensorValue host_tensor = tensor;
        host_tensor.memory_device = cvkit::infer::MemoryDevice::host;
        host_tensor.storage = cvkit::infer::StorageKind::owned;
        host_tensor.external_data = nullptr;
        host_tensor.storage_bytes = 0U;
        host_tensor.storage_owner.reset();
        host_tensor.data.resize(tensor.element_count());

        if (cudaMemcpy(
                host_tensor.data.data(),
                tensor.external_data,
                tensor.packed_byte_size(),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            if (error_message != nullptr)
            {
                *error_message = "failed to copy cuda tensor to host";
            }
            return std::nullopt;
        }

        return host_tensor;
#endif
    }

}  // namespace cvkit::infer::detail
