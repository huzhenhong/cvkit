#include "cvkit/infer/debug.h"

#include <ostream>
#include <sstream>
#include <vector>

namespace cvkit::infer
{

    namespace
    {

        [[nodiscard]] std::string json_escape(std::string_view value)
        {
            std::string result;
            result.reserve(value.size() + 8);
            for (const char ch : value)
            {
                switch (ch)
                {
                    case '\\':
                        result += "\\\\";
                        break;
                    case '"':
                        result += "\\\"";
                        break;
                    case '\n':
                        result += "\\n";
                        break;
                    case '\r':
                        result += "\\r";
                        break;
                    case '\t':
                        result += "\\t";
                        break;
                    default:
                        result += ch;
                        break;
                }
            }
            return result;
        }

        void append_json_string_array(std::ostringstream& stream, const std::vector<std::string>& values)
        {
            stream << '[';
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                if (index > 0)
                {
                    stream << ',';
                }
                stream << '"' << json_escape(values[index]) << '"';
            }
            stream << ']';
        }

        void append_json_tensor_info_array(
            std::ostringstream&            stream,
            const std::vector<TensorInfo>& tensors)
        {
            stream << '[';
            for (std::size_t index = 0; index < tensors.size(); ++index)
            {
                if (index > 0)
                {
                    stream << ',';
                }
                stream << "{";
                stream << "\"name\":\"" << json_escape(tensors[index].name) << "\",";
                stream << "\"shape\":[";
                for (std::size_t dim = 0; dim < tensors[index].shape.size(); ++dim)
                {
                    if (dim > 0)
                    {
                        stream << ',';
                    }
                    stream << tensors[index].shape[dim];
                }
                stream << "],";
                stream << "\"data_type\":\"" << json_escape(data_type_name(tensors[index].data_type)) << "\",";
                stream << "\"memory_device\":\"" << json_escape(memory_device_name(tensors[index].memory_device)) << "\"";
                stream << "}";
            }
            stream << ']';
        }

        void print_session_tensor_list(
            std::ostream&                  stream,
            std::string_view               label,
            const std::vector<TensorInfo>& tensors)
        {
            stream << label;
            for (std::size_t index = 0; index < tensors.size(); ++index)
            {
                if (index > 0)
                {
                    stream << ',';
                }
                stream << tensors[index].name << '(';
                for (std::size_t dim = 0; dim < tensors[index].shape.size(); ++dim)
                {
                    if (dim > 0)
                    {
                        stream << 'x';
                    }
                    stream << tensors[index].shape[dim];
                }
                stream << ';'
                       << "dtype=" << data_type_name(tensors[index].data_type)
                       << ';'
                       << "mem=" << memory_device_name(tensors[index].memory_device)
                       << ')';
            }
            stream << '\n';
        }

    }  // namespace

    std::string_view backend_name(Backend backend)
    {
        switch (backend)
        {
            case Backend::onnxruntime:
                return "onnxruntime";
            case Backend::tensorrt:
                return "tensorrt";
            case Backend::none:
            default:
                return "none";
        }
    }

    std::string_view task_name(TaskKind task)
    {
        switch (task)
        {
            case TaskKind::detection:
                return "detection";
            case TaskKind::classification:
                return "classification";
            case TaskKind::promptable_segmentation:
                return "promptable_segmentation";
            case TaskKind::unknown:
            default:
                return "unknown";
        }
    }

    std::string_view cache_policy_name(CachePolicy policy)
    {
        switch (policy)
        {
            case CachePolicy::disabled:
                return "disabled";
            case CachePolicy::read_only:
                return "read_only";
            case CachePolicy::rebuild:
                return "rebuild";
            case CachePolicy::default_policy:
            default:
                return "default";
        }
    }

    std::string_view data_type_name(TensorDataType data_type)
    {
        switch (data_type)
        {
            case TensorDataType::float16:
                return "float16";
            case TensorDataType::int32:
                return "int32";
            case TensorDataType::int64:
                return "int64";
            case TensorDataType::uint8:
                return "uint8";
            case TensorDataType::boolean:
                return "bool";
            case TensorDataType::float32:
                return "float32";
            case TensorDataType::unknown:
            default:
                return "unknown";
        }
    }

    std::string_view memory_device_name(MemoryDevice device)
    {
        switch (device)
        {
            case MemoryDevice::cuda:
                return "cuda";
            case MemoryDevice::npu:
                return "npu";
            case MemoryDevice::host:
            default:
                return "host";
        }
    }

    std::string_view storage_kind_name(StorageKind storage)
    {
        switch (storage)
        {
            case StorageKind::external_view:
                return "external_view";
            case StorageKind::owned:
            default:
                return "owned";
        }
    }

    void print_graph_info(std::ostream& stream, const Model& model)
    {
        const auto graph   = model.graph_info();
        const auto session = model.session_info();

        stream << "graph.nodes=" << graph.nodes.size() << '\n';
        for (const auto& node : graph.nodes)
        {
            stream << "  node=" << node.name;
            if (!node.depends_on.empty())
            {
                stream << " depends_on=";
                for (std::size_t index = 0; index < node.depends_on.size(); ++index)
                {
                    if (index > 0)
                    {
                        stream << ',';
                    }
                    stream << node.depends_on[index];
                }
            }
            if (!node.consumes.empty())
            {
                stream << " consumes=";
                for (std::size_t index = 0; index < node.consumes.size(); ++index)
                {
                    if (index > 0)
                    {
                        stream << ',';
                    }
                    stream << node.consumes[index];
                }
            }
            if (!node.produces.empty())
            {
                stream << " produces=";
                for (std::size_t index = 0; index < node.produces.size(); ++index)
                {
                    if (index > 0)
                    {
                        stream << ',';
                    }
                    stream << node.produces[index];
                }
            }
            stream << '\n';
        }

        stream << "graph.inputs=";
        for (std::size_t index = 0; index < graph.boundary.inputs.size(); ++index)
        {
            if (index > 0)
            {
                stream << ',';
            }
            stream << graph.boundary.inputs[index];
        }
        stream << '\n';

        stream << "graph.outputs=";
        for (std::size_t index = 0; index < graph.boundary.outputs.size(); ++index)
        {
            if (index > 0)
            {
                stream << ',';
            }
            stream << graph.boundary.outputs[index];
        }
        stream << '\n';

        print_session_tensor_list(stream, "session.inputs=", session.inputs);
        print_session_tensor_list(stream, "session.outputs=", session.outputs);
    }

    void print_graph_trace(std::ostream& stream, const Model& model)
    {
        const auto trace = model.last_graph_trace();
        stream << "graph.trace.nodes=" << trace.size() << '\n';
        for (const auto& node : trace)
        {
            stream
                << "  trace.node=" << node.name
                << " seq=" << node.sequence
                << " duration_us=" << node.duration_us
                << " input_count=" << node.input_count
                << " output_count=" << node.output_count
                << " scratch_count=" << node.scratch_count
                << " ok=" << (node.ok ? "true" : "false");
            if (!node.message.empty())
            {
                stream << " message=\"" << json_escape(node.message) << '"';
            }
            stream << '\n';
        }
    }

    std::string build_graph_json(
        const Model&     model,
        bool             async_infer,
        std::string_view extra_fields_json)
    {
        const auto         graph   = model.graph_info();
        const auto         trace   = model.last_graph_trace();
        const auto         session = model.session_info();

        std::ostringstream stream;
        stream << "{\n";
        stream << "  \"version\": 5,\n";
        stream << "  \"task\": \"" << task_name(model.task()) << "\",\n";
        stream << "  \"backend\": \"" << backend_name(model.backend()) << "\",\n";
        stream << "  \"family\": \"" << json_escape(model.family()) << "\",\n";
        stream << "  \"model_path\": \"" << json_escape(model.model_path()) << "\",\n";
        stream << "  \"labels_path\": \"" << json_escape(model.labels_path()) << "\",\n";
        stream << "  \"cache_policy\": \"" << cache_policy_name(model.cache_policy()) << "\",\n";
        stream << "  \"cache_dir\": \"" << json_escape(model.cache_dir()) << "\",\n";
        stream << "  \"async\": " << (async_infer ? "true" : "false");
        if (!extra_fields_json.empty())
        {
            stream << ",\n"
                   << extra_fields_json;
        }
        else
        {
            stream << ",\n";
        }
        stream << "  \"session\": {\n";
        stream << "    \"inputs\": ";
        append_json_tensor_info_array(stream, session.inputs);
        stream << ",\n";
        stream << "    \"outputs\": ";
        append_json_tensor_info_array(stream, session.outputs);
        stream << "\n";
        stream << "  },\n";
        stream << "  \"nodes\": [\n";
        for (std::size_t index = 0; index < graph.nodes.size(); ++index)
        {
            const auto& node = graph.nodes[index];
            stream << "    {\n";
            stream << "      \"name\": \"" << json_escape(node.name) << "\",\n";
            stream << "      \"depends_on\": ";
            append_json_string_array(stream, node.depends_on);
            stream << ",\n";
            stream << "      \"consumes\": ";
            append_json_string_array(stream, node.consumes);
            stream << ",\n";
            stream << "      \"produces\": ";
            append_json_string_array(stream, node.produces);
            stream << '\n';
            stream << "    }";
            if (index + 1 < graph.nodes.size())
            {
                stream << ',';
            }
            stream << '\n';
        }
        stream << "  ],\n";
        stream << "  \"boundary\": {\n";
        stream << "    \"inputs\": ";
        append_json_string_array(stream, graph.boundary.inputs);
        stream << ",\n";
        stream << "    \"outputs\": ";
        append_json_string_array(stream, graph.boundary.outputs);
        stream << '\n';
        stream << "  },\n";
        stream << "  \"trace\": [\n";
        for (std::size_t index = 0; index < trace.size(); ++index)
        {
            const auto& node = trace[index];
            stream << "    {\n";
            stream << "      \"name\": \"" << json_escape(node.name) << "\",\n";
            stream << "      \"sequence\": " << node.sequence << ",\n";
            stream << "      \"input_count\": " << node.input_count << ",\n";
            stream << "      \"output_count\": " << node.output_count << ",\n";
            stream << "      \"scratch_count\": " << node.scratch_count << ",\n";
            stream << "      \"duration_us\": " << node.duration_us << ",\n";
            stream << "      \"ok\": " << (node.ok ? "true" : "false") << ",\n";
            stream << "      \"message\": \"" << json_escape(node.message) << "\"\n";
            stream << "    }";
            if (index + 1 < trace.size())
            {
                stream << ',';
            }
            stream << '\n';
        }
        stream << "  ]\n";
        stream << "}\n";
        return stream.str();
    }

}  // namespace cvkit::infer
