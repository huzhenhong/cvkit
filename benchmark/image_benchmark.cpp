#include <benchmark/benchmark.h>

#include "cvkit/image/ops.h"

static void bm_resize(benchmark::State& state)
{
    cvkit::core::Frame frame{};
    frame.desc.width = 1920;
    frame.desc.height = 1080;
    frame.desc.channels = 3;
    frame.data.assign(static_cast<std::size_t>(frame.desc.width * frame.desc.height * frame.desc.channels), 0U);

    for (auto _ : state)
    {
        auto output = cvkit::image::resize(frame, 640, 640);
        benchmark::DoNotOptimize(output.data.data());
    }
}

BENCHMARK(bm_resize);
