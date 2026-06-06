#pragma once

#include "cvkit/core/types.h"
#include "cvkit/infer/device.h"
#include "cvkit/infer/infer_export.h"

#include <future>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cvkit::infer
{

    using DetectionListValue = std::vector<cvkit::core::Detection>;
    using PointListValue     = std::vector<cvkit::core::Point2f>;
    using FloatListValue     = std::vector<float>;

    enum class BK_INFER_EXPORT StorageKind : std::uint8_t
    {
        owned,
        external_view,
    };

    struct BK_INFER_EXPORT ImageValue
    {
        cvkit::core::Frame frame{};
        MemoryDevice       memory_device{MemoryDevice::host};
        DeviceRef          device{};
        StorageKind        storage{StorageKind::owned};
        std::size_t        row_stride_bytes{0};
        const void*        external_data{nullptr};
        std::size_t        storage_bytes{0};
        std::shared_ptr<void> storage_owner{};

        [[nodiscard]] bool is_host_accessible() const
        {
            return memory_device == MemoryDevice::host;
        }

        [[nodiscard]] bool owns_storage() const
        {
            return storage == StorageKind::owned;
        }

        [[nodiscard]] std::size_t bytes_per_pixel() const
        {
            if (frame.desc.format == cvkit::core::PixelFormat::nv12)
            {
                return 1U;
            }
            return frame.desc.channels > 0 ? static_cast<std::size_t>(frame.desc.channels) : 0U;
        }

        [[nodiscard]] std::size_t packed_row_stride_bytes() const
        {
            return frame.desc.width > 0 ? static_cast<std::size_t>(frame.desc.width) * bytes_per_pixel() : 0U;
        }

        [[nodiscard]] std::size_t effective_row_stride_bytes() const
        {
            return row_stride_bytes > 0 ? row_stride_bytes : packed_row_stride_bytes();
        }

        [[nodiscard]] bool is_packed() const
        {
            return effective_row_stride_bytes() == packed_row_stride_bytes();
        }

        [[nodiscard]] std::size_t required_byte_size() const
        {
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0)
            {
                return 0U;
            }
            const auto row_stride = effective_row_stride_bytes();
            if (row_stride == 0U)
            {
                return 0U;
            }
            const auto height = static_cast<std::size_t>(frame.desc.height);
            if (frame.desc.format == cvkit::core::PixelFormat::nv12)
            {
                return row_stride * height * 3U / 2U;
            }
            return row_stride * height;
        }

        [[nodiscard]] bool has_valid_host_layout() const
        {
            if (!is_host_accessible())
            {
                return false;
            }
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0 ||
                frame.data.empty())
            {
                return false;
            }
            const auto stride = effective_row_stride_bytes();
            if (stride == 0)
            {
                return false;
            }
            return frame.data.size() >= required_byte_size();
        }

        [[nodiscard]] bool has_valid_device_view() const
        {
            if (is_host_accessible())
            {
                return false;
            }
            if (frame.desc.width <= 0 || frame.desc.height <= 0 || frame.desc.channels <= 0)
            {
                return false;
            }
            if (external_data == nullptr)
            {
                return false;
            }
            const auto required = required_byte_size();
            return required > 0U && storage_bytes >= required;
        }
    };

    inline ImageValue image_value_from_device_frame(const cvkit::core::DeviceFrame& frame)
    {
        ImageValue image{};
        image.frame.desc = frame.desc;
        image.frame.pts = frame.pts;
        image.frame.source = frame.source;
        switch (frame.memory_device)
        {
            case cvkit::core::MemoryDevice::cuda:
                image.memory_device = MemoryDevice::cuda;
                image.device = DeviceRef{DeviceKind::cuda, frame.device_index};
                break;
            case cvkit::core::MemoryDevice::npu:
                image.memory_device = MemoryDevice::npu;
                image.device = DeviceRef{DeviceKind::npu, frame.device_index};
                break;
            case cvkit::core::MemoryDevice::host:
            default:
                image.memory_device = MemoryDevice::host;
                image.device = DeviceRef{DeviceKind::cpu, frame.device_index};
                break;
        }
        image.storage = StorageKind::external_view;
        image.row_stride_bytes = frame.stride_bytes;
        image.external_data = reinterpret_cast<const void*>(frame.data);
        image.storage_bytes = frame.bytes;
        image.storage_owner = frame.owner;
        return image;
    }

    struct BK_INFER_EXPORT MaskValue
    {
        cvkit::core::Frame frame{};
    };

    struct BK_INFER_EXPORT ClassificationValue
    {
        int         class_id{-1};
        float       score{0.0F};
        std::string label{};
    };

    struct BK_INFER_EXPORT BoxListValue
    {
        std::vector<cvkit::core::BBox> boxes{};
    };

    struct BK_INFER_EXPORT KeypointsValue
    {
        std::vector<cvkit::core::Point2f> points{};
    };

    enum class BK_INFER_EXPORT TensorDataType : std::uint8_t
    {
        unknown,
        float32,
        float16,
        int32,
        int64,
        uint8,
        boolean,
    };

    struct BK_INFER_EXPORT TensorValue
    {
        std::string               name{};
        std::vector<std::int64_t> shape{};
        std::vector<float>        data{};
        TensorDataType            data_type{TensorDataType::float32};
        MemoryDevice              memory_device{MemoryDevice::host};
        StorageKind               storage{StorageKind::owned};
        const void*               external_data{nullptr};
        std::size_t               storage_bytes{0};
        std::shared_ptr<void>     storage_owner{};

        [[nodiscard]] bool        is_host_accessible() const
        {
            return memory_device == MemoryDevice::host;
        }

        [[nodiscard]] bool owns_storage() const
        {
            return storage == StorageKind::owned;
        }

        [[nodiscard]] std::size_t element_count() const
        {
            if (shape.empty())
            {
                return 0U;
            }

            std::size_t count = 1U;
            for (const auto dim : shape)
            {
                if (dim <= 0)
                {
                    return 0U;
                }
                count *= static_cast<std::size_t>(dim);
            }
            return count;
        }

        [[nodiscard]] std::size_t packed_byte_size() const
        {
            return element_count() * sizeof(float);
        }

        [[nodiscard]] std::size_t byte_size() const
        {
            return data.size() * sizeof(float);
        }

        [[nodiscard]] bool is_packed() const
        {
            const auto count = element_count();
            return count > 0U && data.size() == count;
        }

        [[nodiscard]] bool has_valid_host_layout() const
        {
            return is_host_accessible() && element_count() > 0U && data.size() >= element_count();
        }

        [[nodiscard]] bool has_valid_device_view() const
        {
            if (is_host_accessible())
            {
                return false;
            }
            const auto required = packed_byte_size();
            return required > 0U && external_data != nullptr && storage_bytes >= required;
        }
    };

    using Value = std::variant<
        ImageValue,
        cvkit::core::Frame,
        MaskValue,
        ClassificationValue,
        DetectionListValue,
        PointListValue,
        cvkit::core::BBox,
        BoxListValue,
        KeypointsValue,
        TensorValue,
        std::string,
        FloatListValue>;

    struct BK_INFER_EXPORT NamedValue
    {
        std::string name{};
        Value       value{};
    };

    struct BK_INFER_EXPORT TaskInput
    {
        std::vector<NamedValue> items{};

        template<typename T>
        [[nodiscard]] const T* find(std::string_view name) const
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                {
                    return std::get_if<T>(&item.value);
                }
            }
            return nullptr;
        }

        template<typename T>
        void add(std::string name, T&& value)
        {
            items.push_back(NamedValue{std::move(name), Value{std::forward<T>(value)}});
        }
    };

    struct BK_INFER_EXPORT TaskOutput
    {
        std::vector<NamedValue> items{};

        template<typename T>
        [[nodiscard]] const T* find(std::string_view name) const
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                {
                    return std::get_if<T>(&item.value);
                }
            }
            return nullptr;
        }

        template<typename T>
        void add(std::string name, T&& value)
        {
            items.push_back(NamedValue{std::move(name), Value{std::forward<T>(value)}});
        }
    };

    class BK_INFER_EXPORT TaskFuture
    {
      public:
        TaskFuture() = default;
        explicit TaskFuture(std::shared_future<TaskOutput> future)
            : future_(std::move(future))
        {
        }

        [[nodiscard]] bool valid() const
        {
            return future_.valid();
        }

        TaskOutput get()
        {
            return future_.get();
        }

        template<typename Rep, typename Period>
        [[nodiscard]] std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeout) const
        {
            return future_.wait_for(timeout);
        }

        [[nodiscard]] bool is_ready() const
        {
            return future_.valid() &&
                   future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }

      private:
        std::shared_future<TaskOutput> future_{};
    };

}  // namespace cvkit::infer
