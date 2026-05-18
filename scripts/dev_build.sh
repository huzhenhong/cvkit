#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"
CONAN_BIN="${CONAN_BIN:-/root/.local/bin/conan}"
CPPSTD="${CPPSTD:-17}"
BUILD_EXAMPLES="${BUILD_EXAMPLES:-ON}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
BUILD_BENCHMARKS="${BUILD_BENCHMARKS:-OFF}"
ENABLE_GSTREAMER="${ENABLE_GSTREAMER:-ON}"
ENABLE_GSTREAMER_CUDA="${ENABLE_GSTREAMER_CUDA:-OFF}"
ENABLE_FFMPEG="${ENABLE_FFMPEG:-OFF}"
ENABLE_ONNXRUNTIME="${ENABLE_ONNXRUNTIME:-ON}"
ENABLE_OPENVINO="${ENABLE_OPENVINO:-OFF}"
ENABLE_TENSORRT="${ENABLE_TENSORRT:-OFF}"
OPENCV_PREFIX="${OPENCV_PREFIX:-}"
ONNXRUNTIME_PREFIX="${ONNXRUNTIME_PREFIX:-}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-}"
OPENVINO_PREFIX="${OPENVINO_PREFIX:-}"
TENSORRT_PREFIX="${TENSORRT_PREFIX:-}"
CUDATOOLKIT_ROOT="${CUDATOOLKIT_ROOT:-}"
CUDART_LIBRARY="${CUDART_LIBRARY:-/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so}"

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/dev_build.sh [release|debug|both]

Default:
  release

Behavior:
  - runs conan install, cmake configure, cmake build, and ctest
  - configures CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  - syncs build/conan/<Config>/compile_commands.json to build/compile_commands.json

Environment overrides:
  CVKIT_DIR
  CONAN_BIN
  CPPSTD
  BUILD_EXAMPLES=ON|OFF
  BUILD_TESTS=ON|OFF
  BUILD_BENCHMARKS=ON|OFF
  ENABLE_GSTREAMER=ON|OFF
  ENABLE_GSTREAMER_CUDA=ON|OFF
  ENABLE_FFMPEG=ON|OFF
  ENABLE_ONNXRUNTIME=ON|OFF
  ENABLE_OPENVINO=ON|OFF
  ENABLE_TENSORRT=ON|OFF
  OPENCV_PREFIX
  ONNXRUNTIME_PREFIX
  FFMPEG_PREFIX
  OPENVINO_PREFIX
  TENSORRT_PREFIX
  CUDATOOLKIT_ROOT
  CUDART_LIBRARY
USAGE
}

require_dir() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        echo "required directory not found: ${dir}" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "required file not found: ${path}" >&2
        exit 1
    fi
}

print_section() {
    printf '\n[%s]\n' "$1"
}

normalize_build_type() {
    local value="$1"
    case "${value}" in
        release | Release)
            printf 'Release'
            ;;
        debug | Debug)
            printf 'Debug'
            ;;
        *)
            echo "unsupported build type: ${value}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

conan_install() {
    local build_type="$1"

    print_section "conan install ${build_type}"
    (
        cd "${CVKIT_DIR}"
        "${CONAN_BIN}" install . \
            -of build \
            -s "compiler.cppstd=${CPPSTD}" \
            -s "build_type=${build_type}" \
            --build=missing
    )
}

cmake_configure() {
    local build_type="$1"
    local build_dir="${CVKIT_DIR}/build/conan/${build_type}"
    local args=(
        -S "${CVKIT_DIR}"
        -B "${build_dir}"
        -DCMAKE_TOOLCHAIN_FILE="${build_dir}/generators/conan_toolchain.cmake"
        -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
        -DCMAKE_BUILD_TYPE="${build_type}"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCVKIT_BUILD_EXAMPLES="${BUILD_EXAMPLES}"
        -DCVKIT_BUILD_TESTS="${BUILD_TESTS}"
        -DCVKIT_BUILD_BENCHMARKS="${BUILD_BENCHMARKS}"
        -DCVKIT_ENABLE_GSTREAMER="${ENABLE_GSTREAMER}"
        -DCVKIT_ENABLE_GSTREAMER_CUDA="${ENABLE_GSTREAMER_CUDA}"
        -DCVKIT_ENABLE_FFMPEG="${ENABLE_FFMPEG}"
        -DCVKIT_ENABLE_ONNXRUNTIME="${ENABLE_ONNXRUNTIME}"
        -DCVKIT_ENABLE_OPENVINO="${ENABLE_OPENVINO}"
        -DCVKIT_ENABLE_TENSORRT="${ENABLE_TENSORRT}"
        -DCVKIT_CUDART_LIBRARY="${CUDART_LIBRARY}"
    )

    if [[ -n "${OPENCV_PREFIX}" ]]; then
        args+=("-DCVKIT_OPENCV_PREFIX=${OPENCV_PREFIX}")
    fi
    if [[ -n "${ONNXRUNTIME_PREFIX}" ]]; then
        args+=("-DCVKIT_ONNXRUNTIME_PREFIX=${ONNXRUNTIME_PREFIX}")
    fi
    if [[ -n "${FFMPEG_PREFIX}" ]]; then
        args+=("-DCVKIT_FFMPEG_PREFIX=${FFMPEG_PREFIX}")
    fi
    if [[ -n "${OPENVINO_PREFIX}" ]]; then
        args+=("-DCVKIT_OPENVINO_PREFIX=${OPENVINO_PREFIX}")
    fi
    if [[ -n "${TENSORRT_PREFIX}" ]]; then
        args+=("-DCVKIT_TENSORRT_PREFIX=${TENSORRT_PREFIX}")
    fi
    if [[ -n "${CUDATOOLKIT_ROOT}" ]]; then
        args+=("-DCVKIT_CUDATOOLKIT_ROOT=${CUDATOOLKIT_ROOT}")
    fi

    print_section "cmake configure ${build_type}"
    cmake "${args[@]}"
}

cmake_build() {
    local build_type="$1"
    print_section "cmake build ${build_type}"
    cmake --build "${CVKIT_DIR}/build/conan/${build_type}"
}

sync_compile_commands() {
    local build_type="$1"
    local source_db="${CVKIT_DIR}/build/conan/${build_type}/compile_commands.json"
    local target_db="${CVKIT_DIR}/build/compile_commands.json"

    if [[ ! -f "${source_db}" ]]; then
        echo "compile_commands.json not found: ${source_db}" >&2
        exit 1
    fi

    print_section "sync compile_commands ${build_type}"
    mkdir -p "$(dirname "${target_db}")"
    cp "${source_db}" "${target_db}"
}

cmake_test() {
    local build_type="$1"

    if [[ "${BUILD_TESTS}" != "ON" ]]; then
        return
    fi

    print_section "ctest ${build_type}"
    ctest --test-dir "${CVKIT_DIR}/build/conan/${build_type}" --output-on-failure
}

build_one() {
    local build_type="$1"
    conan_install "${build_type}"
    cmake_configure "${build_type}"
    cmake_build "${build_type}"
    sync_compile_commands "${build_type}"
    cmake_test "${build_type}"
}

main() {
    local mode="${1:-release}"

    if [[ "${mode}" == "-h" || "${mode}" == "--help" ]]; then
        usage
        exit 0
    fi

    require_dir "${CVKIT_DIR}"
    require_file "${CONAN_BIN}"

    case "${mode}" in
        release | Release | debug | Debug)
            build_one "$(normalize_build_type "${mode}")"
            ;;
        both)
            build_one Release
            build_one Debug
            ;;
        *)
            echo "unsupported mode: ${mode}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
