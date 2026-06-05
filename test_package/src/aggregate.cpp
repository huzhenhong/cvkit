#include "cvkit/core/types.h"
#include "cvkit/image/ops.h"
#include "cvkit/infer/model.h"
#include "cvkit/media/source.h"

#include <cstddef>
#include <string>

int main()
{
    cvkit::core::Frame frame{};
    frame.desc.width    = 2;
    frame.desc.height   = 2;
    frame.desc.channels = 3;
    frame.desc.format   = cvkit::core::PixelFormat::bgr8;
    frame.data.assign(static_cast<std::size_t>(2 * 2 * 3), 3U);

    const auto resized = cvkit::image::resize(frame, 1, 1);
    if (resized.desc.width != 1 || resized.desc.height != 1)
    {
        return 1;
    }

    cvkit::infer::Model model;
    if (model.loaded())
    {
        return 1;
    }

    cvkit::media::Source source;
    return source.open(std::string{}) ? 1 : 0;
}
