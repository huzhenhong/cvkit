#include "graph.h"

#include <basekit/log/logger.h>

#include "../tasks/detection/detection_pipeline.h"
#include "../tasks/promptable_segmentation/promptable_segmentation_pipeline.h"
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

#if __has_include(<tracy/Tracy.hpp>)
    #include <tracy/Tracy.hpp>
#else
    #define ZoneScoped ((void)0)
    #define ZoneName(name, size) ((void)0)
#endif

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] bool graph_trace_logging_enabled()
        {
            if (const char* value = std::getenv("CVKIT_TRACE_GRAPH"); value != nullptr)
            {
                return value[0] != '\0' && value[0] != '0';
            }
            return false;
        }

        void log_node_trace(const NodeTrace& trace)
        {
            if (!graph_trace_logging_enabled())
            {
                return;
            }

            static basekit::log::Logger logger{"cvkit_graph"};
            std::ostringstream          stream;
            stream
                << "node=" << trace.name
                << " seq=" << trace.sequence
                << " duration_us=" << trace.duration_us
                << " input_count=" << trace.input_count
                << " output_delta=" << trace.output_count
                << " scratch_delta=" << trace.scratch_count
                << " ok=" << (trace.ok ? "true" : "false");
            if (!trace.message.empty())
            {
                stream << " message=" << trace.message;
            }
            logger.info(stream.str());
        }

        void log_graph_error(std::string_view message)
        {
            static basekit::log::Logger logger{"cvkit_graph"};
            logger.error(message);
        }

        [[nodiscard]] Packet apply_node_with_trace(
            const INode& node,
            Packet       packet,
            std::size_t  sequence)
        {
            const auto before_out = packet.output.items.size();
            const auto before_scratch = packet.scratch.size();
            ZoneScoped;
            ZoneName(node.name().data(), static_cast<int>(node.name().size()));
            const auto started = std::chrono::steady_clock::now();
            packet = node.process(std::move(packet));
            const auto finished = std::chrono::steady_clock::now();
            const auto duration_us =
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count());
            const auto message = node.trace_message(packet);

            auto trace = NodeTrace{
                std::string{node.name()},
                sequence,
                packet.input.items.size(),
                packet.output.items.size() >= before_out ? packet.output.items.size() - before_out : 0,
                packet.scratch.size() >= before_scratch ? packet.scratch.size() - before_scratch : 0,
                duration_us,
                message.empty(),
                message};
            log_node_trace(trace);
            packet.add_trace(std::move(trace));
            return packet;
        }

        [[nodiscard]] Packet apply_async_node_with_trace(
            const INode& node,
            Packet       packet,
            std::size_t  sequence)
        {
            const auto before_out = packet.output.items.size();
            const auto before_scratch = packet.scratch.size();
            ZoneScoped;
            ZoneName(node.name().data(), static_cast<int>(node.name().size()));
            const auto started = std::chrono::steady_clock::now();
            packet = node.submit(std::move(packet)).get();
            const auto finished = std::chrono::steady_clock::now();
            const auto duration_us =
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count());
            const auto message = node.trace_message(packet);

            auto trace = NodeTrace{
                std::string{node.name()},
                sequence,
                packet.input.items.size(),
                packet.output.items.size() >= before_out ? packet.output.items.size() - before_out : 0,
                packet.scratch.size() >= before_scratch ? packet.scratch.size() - before_scratch : 0,
                duration_us,
                message.empty(),
                message};
            log_node_trace(trace);
            packet.add_trace(std::move(trace));
            return packet;
        }

        class PipelineNode final : public INode
        {
          public:
            PipelineNode(
                std::shared_ptr<IBackendSession> backend,
                std::shared_ptr<ITaskPipeline>   pipeline,
                PipelineContext                  context)
                : backend_(std::move(backend))
                , pipeline_(std::move(pipeline))
                , context_(std::move(context))
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "task_pipeline";
            }

            [[nodiscard]] std::vector<std::string> consumes() const override
            {
                return {};
            }

            [[nodiscard]] std::vector<std::string> produces() const override
            {
                return {};
            }

            [[nodiscard]] Packet process(Packet packet) const override
            {
                packet.output = pipeline_->run_sync(*backend_, packet.input, context_);
                return packet;
            }

          private:
            std::shared_ptr<IBackendSession> backend_{};
            std::shared_ptr<ITaskPipeline>   pipeline_{};
            PipelineContext                  context_{};
        };

        class DetectionInferNode final : public INode
        {
          public:
            explicit DetectionInferNode(std::shared_ptr<IBackendSession> backend)
                : backend_(std::move(backend))
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "detection_infer";
            }

            [[nodiscard]] std::vector<std::string> consumes() const override
            {
                return {"input:image"};
            }

            [[nodiscard]] std::vector<std::string> produces() const override
            {
                return {"scratch:detection.preprocess", "scratch:detection.raw_outputs"};
            }

            [[nodiscard]] bool supports_async() const override
            {
                return backend_ != nullptr && backend_->supports_async();
            }

            [[nodiscard]] std::string trace_message(const Packet& packet) const override
            {
                if (const auto* error = packet.get<std::string>("detection.preprocess_error"); error != nullptr)
                {
                    return *error;
                }
                return {};
            }

            [[nodiscard]] PacketFuture submit(Packet packet) const override
            {
                if (backend_ == nullptr)
                {
                    return make_ready_packet_future(std::move(packet));
                }

                BackendFuture backend_future{};
                if (!prepare_detection_inference_async(*backend_, packet.input, packet, backend_future))
                {
                    return make_ready_packet_future(std::move(packet));
                }

                auto future = std::async(
                                  std::launch::async,
                                  [packet = std::move(packet), backend_future = std::move(backend_future)]() mutable
                                  {
                                      packet.put("detection.raw_outputs", backend_future.get());
                                      return packet;
                                  })
                                  .share();
                return PacketFuture{std::move(future)};
            }

            [[nodiscard]] Packet process(Packet packet) const override
            {
                if (!prepare_detection_inference(*backend_, packet.input, packet))
                {
                    return packet;
                }
                return packet;
            }

          private:
            std::shared_ptr<IBackendSession> backend_{};
        };

        class DetectionPostprocessNode final : public INode
        {
          public:
            explicit DetectionPostprocessNode(PipelineContext context)
                : context_(std::move(context))
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "detection_postprocess";
            }

            [[nodiscard]] std::vector<std::string> consumes() const override
            {
                return {
                    "input:image",
                    "scratch:detection.preprocess",
                    "scratch:detection.raw_outputs"};
            }

            [[nodiscard]] std::vector<std::string> produces() const override
            {
                return {"output:detections"};
            }

            [[nodiscard]] Packet process(Packet packet) const override
            {
                packet.output.add("detections", finalize_detection_output(packet, context_));
                return packet;
            }

          private:
            PipelineContext context_{};
        };

        class PromptableSegmentationInferNode final : public INode
        {
          public:
            PromptableSegmentationInferNode(
                std::shared_ptr<IBackendSession> backend,
                PipelineContext                  context)
                : backend_(std::move(backend))
                , context_(std::move(context))
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "promptable_segmentation_infer";
            }

            [[nodiscard]] std::vector<std::string> consumes() const override
            {
                return {
                    "input:image",
                    "input:image_embeddings",
                    "input:points",
                    "input:point_labels",
                    "input:box",
                    "input:prompt"};
            }

            [[nodiscard]] std::vector<std::string> produces() const override
            {
                return {
                    "scratch:promptable.image_embeddings",
                    "scratch:promptable.mask",
                    "scratch:promptable.low_res_masks",
                    "scratch:promptable.logits",
                    "scratch:promptable.scores",
                    "scratch:promptable.ready"};
            }

            [[nodiscard]] bool supports_async() const override
            {
                return backend_ != nullptr;
            }

            [[nodiscard]] PacketFuture submit(Packet packet) const override
            {
                if (backend_ == nullptr)
                {
                    return make_ready_packet_future(std::move(packet));
                }

                auto future = std::async(
                                  std::launch::async,
                                  [backend = backend_, context = context_, packet = std::move(packet)]() mutable
                                  {
                                      static_cast<void>(prepare_promptable_segmentation_inference(
                                          *backend,
                                          packet.input,
                                          context,
                                          packet));
                                      return packet;
                                  })
                                  .share();
                return PacketFuture{std::move(future)};
            }

            [[nodiscard]] Packet process(Packet packet) const override
            {
                static_cast<void>(prepare_promptable_segmentation_inference(*backend_, packet.input, context_, packet));
                return packet;
            }

          private:
            std::shared_ptr<IBackendSession> backend_{};
            PipelineContext                  context_{};
        };

        class PromptableSegmentationPostprocessNode final : public INode
        {
          public:
            explicit PromptableSegmentationPostprocessNode(PipelineContext context)
                : context_(std::move(context))
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "promptable_segmentation_postprocess";
            }

            [[nodiscard]] std::vector<std::string> consumes() const override
            {
                return {
                    "scratch:promptable.ready",
                    "scratch:promptable.image_embeddings",
                    "scratch:promptable.mask",
                    "scratch:promptable.low_res_masks",
                    "scratch:promptable.logits",
                    "scratch:promptable.scores"};
            }

            [[nodiscard]] std::vector<std::string> produces() const override
            {
                return {
                    "output:image_embeddings",
                    "output:mask",
                    "output:low_res_masks",
                    "output:logits",
                    "output:scores"};
            }

            [[nodiscard]] Packet process(Packet packet) const override
            {
                packet.output = finalize_promptable_segmentation_output(packet, context_);
                return packet;
            }

          private:
            PipelineContext context_{};
        };

    }  // namespace

    void TaskGraph::add_node(std::shared_ptr<INode> node)
    {
        if (node != nullptr)
        {
            std::vector<std::string> depends_on;
            if (!nodes_.empty() && nodes_.back().node != nullptr)
            {
                depends_on.push_back(std::string{nodes_.back().node->name()});
            }
            nodes_.push_back(NodeEntry{std::move(node), std::move(depends_on)});
        }
    }

    void TaskGraph::add_node(std::shared_ptr<INode> node, std::vector<std::string> depends_on)
    {
        if (node != nullptr)
        {
            nodes_.push_back(NodeEntry{std::move(node), std::move(depends_on)});
        }
    }

    std::size_t TaskGraph::node_count() const
    {
        return nodes_.size();
    }

    std::vector<NodeMetadata> TaskGraph::metadata() const
    {
        std::vector<NodeMetadata> nodes;
        nodes.reserve(nodes_.size());
        for (const auto& node : nodes_)
        {
            if (node.node != nullptr)
            {
                nodes.push_back(NodeMetadata{
                    std::string{node.node->name()},
                    node.depends_on,
                    node.node->consumes(),
                    node.node->produces()});
            }
        }
        return nodes;
    }

    GraphBoundary TaskGraph::boundary() const
    {
        std::set<std::string> all_consumed;
        std::set<std::string> all_produced;
        std::set<std::string> external_inputs;
        std::set<std::string> external_outputs;

        for (const auto& entry : nodes_)
        {
            if (entry.node == nullptr)
            {
                continue;
            }

            for (const auto& consumed : entry.node->consumes())
            {
                all_consumed.insert(consumed);
                if (consumed.rfind("input:", 0) == 0)
                {
                    external_inputs.insert(consumed);
                }
            }

            for (const auto& produced : entry.node->produces())
            {
                all_produced.insert(produced);
            }
        }

        for (const auto& produced : all_produced)
        {
            if (produced.rfind("output:", 0) == 0 || all_consumed.find(produced) == all_consumed.end())
            {
                external_outputs.insert(produced);
            }
        }

        GraphBoundary result{};
        result.inputs.assign(external_inputs.begin(), external_inputs.end());
        result.outputs.assign(external_outputs.begin(), external_outputs.end());
        return result;
    }

    std::vector<std::size_t> TaskGraph::execution_order() const
    {
        std::vector<std::size_t> order;
        order.reserve(nodes_.size());

        std::unordered_map<std::string, std::size_t> name_to_index;
        name_to_index.reserve(nodes_.size());
        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            if (nodes_[index].node == nullptr)
            {
                continue;
            }
            name_to_index.emplace(std::string{nodes_[index].node->name()}, index);
        }

        std::vector<std::vector<std::size_t>> edges(nodes_.size());
        std::vector<std::size_t> indegree(nodes_.size(), 0);
        std::unordered_map<std::string, std::size_t> produced_by;
        produced_by.reserve(nodes_.size() * 2);

        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            if (nodes_[index].node == nullptr)
            {
                continue;
            }

            for (const auto& produced_key : nodes_[index].node->produces())
            {
                const auto [_, inserted] = produced_by.emplace(produced_key, index);
                if (!inserted)
                {
                    log_graph_error(std::string{"graph output key produced more than once: "} + produced_key);
                    return {};
                }
            }
        }

        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            if (nodes_[index].node == nullptr)
            {
                continue;
            }

            for (const auto& dependency_name : nodes_[index].depends_on)
            {
                const auto iter = name_to_index.find(dependency_name);
                if (iter == name_to_index.end())
                {
                    log_graph_error(std::string{"graph dependency not found: "} + dependency_name);
                    return {};
                }
                edges[iter->second].push_back(index);
                ++indegree[index];
            }

            for (const auto& consumed_key : nodes_[index].node->consumes())
            {
                if (consumed_key.rfind("input:", 0) == 0)
                {
                    continue;
                }

                const auto producer = produced_by.find(consumed_key);
                if (producer == produced_by.end())
                {
                    continue;
                }

                const auto producer_index = producer->second;
                if (producer_index == index)
                {
                    continue;
                }

                edges[producer_index].push_back(index);
                ++indegree[index];
            }
        }

        std::queue<std::size_t> ready;
        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            if (nodes_[index].node != nullptr && indegree[index] == 0)
            {
                ready.push(index);
            }
        }

        while (!ready.empty())
        {
            const auto current = ready.front();
            ready.pop();
            order.push_back(current);
            for (const auto next : edges[current])
            {
                if (--indegree[next] == 0)
                {
                    ready.push(next);
                }
            }
        }

        std::size_t real_nodes = 0;
        for (const auto& entry : nodes_)
        {
            if (entry.node != nullptr)
            {
                ++real_nodes;
            }
        }

        if (order.size() != real_nodes)
        {
            log_graph_error("graph contains a dependency cycle");
            return {};
        }

        return order;
    }

    Packet TaskGraph::run_sync(Packet packet) const
    {
        ZoneScoped;
        const auto order = execution_order();
        if (order.empty() && !nodes_.empty())
        {
            return packet;
        }

        for (std::size_t sequence = 0; sequence < order.size(); ++sequence)
        {
            const auto  index      = order[sequence];
            const auto& node_entry = nodes_[index];
            const auto& node       = node_entry.node;
            if (node == nullptr)
            {
                continue;
            }
            packet = apply_node_with_trace(*node, std::move(packet), sequence);
        }
        return packet;
    }

    PacketFuture TaskGraph::submit_packet(Packet packet) const
    {
        auto order = execution_order();
        if (order.empty() && !nodes_.empty())
        {
            return {};
        }

        auto packet_future = make_ready_packet_future(std::move(packet));
        for (const auto index : order)
        {
            if (index >= nodes_.size())
            {
                continue;
            }

            auto node = nodes_[index].node;
            if (node == nullptr)
            {
                continue;
            }

            auto previous = std::move(packet_future);
            auto future = std::async(
                              std::launch::async,
                              [node = std::move(node), previous = std::move(previous), sequence = index]() mutable
                              {
                                  auto packet = previous.get();
                                  if (node->supports_async())
                                  {
                                      return apply_async_node_with_trace(*node, std::move(packet), sequence);
                                  }
                                  return apply_node_with_trace(*node, std::move(packet), sequence);
                              })
                              .share();
            packet_future = PacketFuture{std::move(future)};
        }

        return packet_future;
    }

    TaskGraph create_pipeline_graph(
        std::shared_ptr<IBackendSession> backend,
        std::shared_ptr<ITaskPipeline>   pipeline,
        PipelineContext                  context)
    {
        TaskGraph graph{};
        if (pipeline != nullptr && pipeline->task() == TaskKind::detection)
        {
            graph.add_node(std::make_shared<DetectionInferNode>(backend));
            graph.add_node(std::make_shared<DetectionPostprocessNode>(std::move(context)));
            return graph;
        }

        if (pipeline != nullptr && pipeline->task() == TaskKind::promptable_segmentation)
        {
            graph.add_node(std::make_shared<PromptableSegmentationInferNode>(backend, context));
            graph.add_node(std::make_shared<PromptableSegmentationPostprocessNode>(std::move(context)));
            return graph;
        }

        graph.add_node(std::make_shared<PipelineNode>(std::move(backend), std::move(pipeline), std::move(context)));
        return graph;
    }

}  // namespace cvkit::infer::detail
