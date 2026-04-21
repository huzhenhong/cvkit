#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"

rm -rf "${CVKIT_DIR}/build"
rm -f "${CVKIT_DIR}/CMakeUserPresets.json"
rm -rf "${CVKIT_DIR}/test_package/build"
rm -f "${CVKIT_DIR}/test_package/CMakeUserPresets.json"
rm -f "${CVKIT_DIR}/test_package/CMakePresets.json"
