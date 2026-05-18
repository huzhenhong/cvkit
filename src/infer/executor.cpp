#include "executor.h"

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cvkit::infer::detail
{

    namespace
    {

        [[nodiscard]] std::size_t default_thread_count()
        {
            if (const char* raw = std::getenv("CVKIT_EXECUTOR_THREADS"); raw != nullptr)
            {
                try
                {
                    const auto parsed = static_cast<std::size_t>(std::stoul(raw));
                    if (parsed > 0)
                    {
                        return parsed;
                    }
                }
                catch (...)
                {
                }
            }

            const auto detected = std::thread::hardware_concurrency();
            if (detected == 0)
            {
                return 2;
            }
            return detected;
        }

        class ThreadPoolExecutor final : public IExecutor
        {
          public:
            explicit ThreadPoolExecutor(std::size_t thread_count)
            {
                workers_.reserve(thread_count);
                for (std::size_t index = 0; index < thread_count; ++index)
                {
                    workers_.emplace_back([this]()
                                          { worker_loop(); });
                }
            }

            ~ThreadPoolExecutor() override
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                condition_.notify_all();
                for (auto& worker : workers_)
                {
                    if (worker.joinable())
                    {
                        worker.join();
                    }
                }
            }

            [[nodiscard]] TaskFuture submit(std::function<TaskOutput()> task) override
            {
                auto packaged = std::make_shared<std::packaged_task<TaskOutput()>>(std::move(task));
                auto future   = packaged->get_future().share();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stopping_)
                    {
                        std::promise<TaskOutput> promise;
                        auto                     ready = promise.get_future().share();
                        promise.set_value(TaskOutput{});
                        return TaskFuture{std::move(ready)};
                    }

                    queue_.emplace_back([packaged]()
                                        { (*packaged)(); });
                }
                condition_.notify_one();
                return TaskFuture{std::move(future)};
            }

          private:
            void worker_loop()
            {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        condition_.wait(lock, [this]()
                                        { return stopping_ || !queue_.empty(); });
                        if (stopping_ && queue_.empty())
                        {
                            return;
                        }

                        task = std::move(queue_.front());
                        queue_.pop_front();
                    }

                    task();
                }
            }

            std::mutex                        mutex_{};
            std::condition_variable           condition_{};
            std::deque<std::function<void()>> queue_{};
            std::vector<std::thread>          workers_{};
            bool                              stopping_{false};
        };

    }  // namespace

    std::shared_ptr<IExecutor> create_default_executor()
    {
        static std::shared_ptr<IExecutor> executor = std::make_shared<ThreadPoolExecutor>(default_thread_count());
        return executor;
    }

}  // namespace cvkit::infer::detail
