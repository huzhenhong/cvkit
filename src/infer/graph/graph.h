#pragma once

#include "cvkit/infer/infer_export.h"

#include "../backends/backend_session.h"
#include "../executor.h"
#include "../tasks/task_pipeline.h"
#include "packet.h"

#include <memory>
#include <chrono>
#include <future>
#include <string_view>
#include <string>
#include <vector>

namespace cvkit::infer::detail
{

    class PacketFuture
    {
      public:
        PacketFuture() = default;
        explicit PacketFuture(std::shared_future<Packet> future)
            : future_(std::move(future))
        {
        }

        [[nodiscard]] bool valid() const
        {
            return future_.valid();
        }

        Packet get()
        {
            return future_.get();
        }

        template<typename Rep, typename Period>
        [[nodiscard]] std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeout) const
        {
            return future_.wait_for(timeout);
        }

      private:
        std::shared_future<Packet> future_{};
    };

    [[nodiscard]] inline PacketFuture make_ready_packet_future(Packet packet)
    {
        std::promise<Packet> promise;
        auto                 future = promise.get_future().share();
        promise.set_value(std::move(packet));
        return PacketFuture{std::move(future)};
    }

    class INode
    {
      public:
        virtual ~INode() = default;

        virtual std::string_view         name() const = 0;
        virtual std::vector<std::string> consumes() const
        {
            return {};
        }
        virtual std::vector<std::string> produces() const
        {
            return {};
        }
        virtual bool supports_async() const
        {
            return false;
        }
        virtual std::string trace_message(const Packet& packet) const
        {
            static_cast<void>(packet);
            return {};
        }
        virtual PacketFuture submit(Packet packet) const
        {
            return make_ready_packet_future(process(std::move(packet)));
        }
        virtual Packet process(Packet packet) const = 0;
    };

    struct NodeMetadata
    {
        std::string              name{};
        std::vector<std::string> depends_on{};
        std::vector<std::string> consumes{};
        std::vector<std::string> produces{};
    };

    struct GraphBoundary
    {
        std::vector<std::string> inputs{};
        std::vector<std::string> outputs{};
    };

    class BK_INFER_EXPORT TaskGraph
    {
      public:
        void                       add_node(std::shared_ptr<INode> node);
        void                       add_node(std::shared_ptr<INode> node, std::vector<std::string> depends_on);
        std::size_t                node_count() const;
        std::vector<NodeMetadata>  metadata() const;
        GraphBoundary              boundary() const;
        [[nodiscard]] Packet       run_sync(Packet packet) const;
        [[nodiscard]] PacketFuture submit_packet(Packet packet) const;

      private:
        struct NodeEntry
        {
            std::shared_ptr<INode>   node{};
            std::vector<std::string> depends_on{};
        };

        std::vector<std::size_t> execution_order() const;

        std::vector<NodeEntry>   nodes_{};
    };

    TaskGraph create_pipeline_graph(
        std::shared_ptr<IBackendSession> backend,
        std::shared_ptr<ITaskPipeline>   pipeline,
        PipelineContext                  context);

}  // namespace cvkit::infer::detail
