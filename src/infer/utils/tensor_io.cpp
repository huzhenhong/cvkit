#include "cvkit/infer/tensor_io.h"

#include <cstdint>
#include <fstream>
#include <string_view>
#include <system_error>

namespace cvkit::infer
{

    namespace
    {

        struct TensorFileHeaderV1
        {
            char          magic[8]{'C', 'V', 'K', 'T', 'E', 'M', 'B', '1'};
            std::uint32_t name_size{0};
            std::uint32_t rank{0};
            std::uint64_t value_count{0};
        };

        struct TensorFileHeaderV2
        {
            char          magic[8]{'C', 'V', 'K', 'T', 'E', 'M', 'B', '2'};
            std::uint32_t name_size{0};
            std::uint32_t rank{0};
            std::uint64_t value_count{0};
            std::uint32_t data_type{0};
            std::uint32_t memory_device{0};
        };

        struct TensorFileHeaderV3
        {
            char          magic[8]{'C', 'V', 'K', 'T', 'E', 'M', 'B', '3'};
            std::uint32_t name_size{0};
            std::uint32_t rank{0};
            std::uint64_t value_count{0};
            std::uint32_t data_type{0};
            std::uint32_t memory_device{0};
            std::uint32_t storage{0};
        };

    }  // namespace

    bool save_tensor_file(
        const TensorValue&           tensor,
        const std::filesystem::path& path)
    {
        if (tensor.data.empty())
        {
            return false;
        }

        std::error_code ec;
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path(), ec);
        }

        std::ofstream output(path, std::ios::binary);
        if (!output.is_open())
        {
            return false;
        }

        TensorFileHeaderV3 header{};
        header.name_size     = static_cast<std::uint32_t>(tensor.name.size());
        header.rank          = static_cast<std::uint32_t>(tensor.shape.size());
        header.value_count   = static_cast<std::uint64_t>(tensor.data.size());
        header.data_type     = static_cast<std::uint32_t>(tensor.data_type);
        header.memory_device = static_cast<std::uint32_t>(tensor.memory_device);
        header.storage       = static_cast<std::uint32_t>(tensor.storage);

        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
        output.write(
            reinterpret_cast<const char*>(tensor.shape.data()),
            static_cast<std::streamsize>(tensor.shape.size() * sizeof(std::int64_t)));
        output.write(
            reinterpret_cast<const char*>(tensor.data.data()),
            static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
        return output.good();
    }

    bool load_tensor_file(
        const std::filesystem::path& path,
        TensorValue&                 tensor)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input.is_open())
        {
            return false;
        }

        char magic[8]{};
        input.read(magic, sizeof(magic));
        if (!input.good())
        {
            return false;
        }

        input.seekg(0, std::ios::beg);

        const auto magic_view = std::string_view(magic, sizeof(magic));
        if (magic_view == "CVKTEMB3")
        {
            TensorFileHeaderV3 header{};
            input.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!input.good())
            {
                return false;
            }

            tensor.name.resize(header.name_size);
            tensor.shape.resize(header.rank);
            tensor.data.resize(static_cast<std::size_t>(header.value_count));
            tensor.data_type     = static_cast<TensorDataType>(header.data_type);
            tensor.memory_device = static_cast<MemoryDevice>(header.memory_device);
            tensor.storage       = static_cast<StorageKind>(header.storage);

            input.read(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
            input.read(
                reinterpret_cast<char*>(tensor.shape.data()),
                static_cast<std::streamsize>(tensor.shape.size() * sizeof(std::int64_t)));
            input.read(
                reinterpret_cast<char*>(tensor.data.data()),
                static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
            return input.good();
        }

        if (magic_view == "CVKTEMB2")
        {
            TensorFileHeaderV2 header{};
            input.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!input.good())
            {
                return false;
            }

            tensor.name.resize(header.name_size);
            tensor.shape.resize(header.rank);
            tensor.data.resize(static_cast<std::size_t>(header.value_count));
            tensor.data_type     = static_cast<TensorDataType>(header.data_type);
            tensor.memory_device = static_cast<MemoryDevice>(header.memory_device);
            tensor.storage       = StorageKind::owned;

            input.read(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
            input.read(
                reinterpret_cast<char*>(tensor.shape.data()),
                static_cast<std::streamsize>(tensor.shape.size() * sizeof(std::int64_t)));
            input.read(
                reinterpret_cast<char*>(tensor.data.data()),
                static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
            return input.good();
        }

        if (magic_view != "CVKTEMB1")
        {
            return false;
        }

        TensorFileHeaderV1 header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input.good())
        {
            return false;
        }

        tensor.name.resize(header.name_size);
        tensor.shape.resize(header.rank);
        tensor.data.resize(static_cast<std::size_t>(header.value_count));
        tensor.data_type     = TensorDataType::float32;
        tensor.memory_device = MemoryDevice::host;
        tensor.storage       = StorageKind::owned;

        input.read(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
        input.read(
            reinterpret_cast<char*>(tensor.shape.data()),
            static_cast<std::streamsize>(tensor.shape.size() * sizeof(std::int64_t)));
        input.read(
            reinterpret_cast<char*>(tensor.data.data()),
            static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
        return input.good();
    }

}  // namespace cvkit::infer
