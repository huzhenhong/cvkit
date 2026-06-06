#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ensure_conan
require_dir "${CVKIT_DIR}"
ensure_basekit_package
conan_install_release
cmake_configure_release
cmake_build_release
ctest_release
run_pipeline_image
run_media_writer_probe
