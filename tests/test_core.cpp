#include <catch2/catch_test_macros.hpp>

#include "cvkit/core/types.h"

TEST_CASE("frame stores image metadata")
{
    cvkit::core::Frame frame{};
    frame.desc.width = 640;
    frame.desc.height = 480;
    frame.desc.channels = 3;

    CHECK(frame.desc.width == 640);
    CHECK(frame.desc.height == 480);
    CHECK(frame.desc.channels == 3);
}
