#pragma once

#include "cvkit/infer/model.h"
#include "cvkit/media/options.h"

#include <filesystem>
#include <string_view>

namespace cvkit::examples
{

    [[nodiscard]] cvkit::media::ReaderBackend parse_reader_backend(std::string_view value);
    [[nodiscard]] cvkit::media::WriterBackend parse_writer_backend(std::string_view value);
    [[nodiscard]] cvkit::media::GstVideoCodec parse_gst_codec(std::string_view value);

    int                                       run_image(
        cvkit::infer::Model&         model,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_dir,
        cvkit::media::ReaderBackend  reader_backend,
        bool                         async_infer,
        bool                         print_graph,
        const std::filesystem::path& dump_graph_json_path);

    int run_video(
        cvkit::infer::Model&         model,
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir,
        std::size_t                  max_frames,
        cvkit::media::ReaderBackend  reader_backend,
        cvkit::media::WriterBackend  writer_backend,
        cvkit::media::GstVideoCodec  gst_codec,
        bool                         async_infer,
        bool                         print_graph,
        const std::filesystem::path& dump_graph_json_path);

}  // namespace cvkit::examples
