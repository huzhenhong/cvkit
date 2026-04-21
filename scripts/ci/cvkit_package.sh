#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ensure_conan
require_dir "${CVKIT_DIR}"
ensure_basekit_package

if [[ -z "$(detect_opencv_prefix)" || -z "$(detect_onnxruntime_prefix)" ]]; then
    echo "Skipping cvkit package CI: OpenCV or ONNX Runtime prefix not detected"
    exit 0
fi

print_section "cvkit create"
(
    cd "${CVKIT_DIR}"
    "${CONAN_BIN}" create . --build=missing -s "compiler.cppstd=${CPPSTD}"
)
