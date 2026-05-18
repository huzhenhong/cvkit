#pragma once

#include "cvkit/infer/infer_export.h"
#include "cvkit/infer/model.h"

#include <iosfwd>
#include <string>
#include <string_view>

namespace cvkit::infer
{

    [[nodiscard]] BK_INFER_EXPORT std::string_view backend_name(Backend backend);
    [[nodiscard]] BK_INFER_EXPORT std::string_view task_name(TaskKind task);
    [[nodiscard]] BK_INFER_EXPORT std::string_view cache_policy_name(CachePolicy policy);
    [[nodiscard]] BK_INFER_EXPORT std::string_view data_type_name(TensorDataType data_type);
    [[nodiscard]] BK_INFER_EXPORT std::string_view memory_device_name(MemoryDevice device);
    [[nodiscard]] BK_INFER_EXPORT std::string_view storage_kind_name(StorageKind storage);

    BK_INFER_EXPORT void                           print_graph_info(std::ostream& stream, const Model& model);
    BK_INFER_EXPORT void                           print_graph_trace(std::ostream& stream, const Model& model);

    [[nodiscard]] BK_INFER_EXPORT std::string build_graph_json(
        const Model&     model,
        bool             async_infer,
        std::string_view extra_fields_json = {});

}  // namespace cvkit::infer
