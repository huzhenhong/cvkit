#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"
BASEKIT_DIR="${BASEKIT_DIR:-/workspace/basekit}"
CONAN_BIN="${CONAN_BIN:-/root/.local/bin/conan}"
CPPSTD="${CPPSTD:-17}"

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

ensure_conan() {
    require_file "${CONAN_BIN}"
}

have_gstreamer() {
    command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0
}

detect_opencv_prefix() {
    if [[ -n "${CVKIT_OPENCV_PREFIX:-}" ]]; then
        printf '%s' "${CVKIT_OPENCV_PREFIX}"
    elif [[ -d /usr/local/lib/cmake/opencv4 ]]; then
        printf '%s' /usr/local/lib/cmake/opencv4
    elif [[ -d /usr/lib/x86_64-linux-gnu/cmake/opencv4 ]]; then
        printf '%s' /usr/lib/x86_64-linux-gnu/cmake/opencv4
    fi
}

detect_onnxruntime_prefix() {
    if [[ -n "${CVKIT_ONNXRUNTIME_PREFIX:-}" ]]; then
        printf '%s' "${CVKIT_ONNXRUNTIME_PREFIX}"
    elif [[ -d /usr/local/lib/cmake/onnxruntime ]]; then
        printf '%s' /usr/local/lib/cmake/onnxruntime
    fi
}

ensure_basekit_package() {
    if [[ -d "${BASEKIT_DIR}" ]]; then
        print_section "basekit create"
        (
            cd "${BASEKIT_DIR}"
            "${CONAN_BIN}" create . --build=missing -s "compiler.cppstd=${CPPSTD}"
        )
    fi
}

conan_install_release() {
    print_section "cvkit conan install"
    (
        cd "${CVKIT_DIR}"
        "${CONAN_BIN}" install . -of build -s "compiler.cppstd=${CPPSTD}" --build=missing
    )
}

cmake_configure_release() {
    local opencv_prefix
    local onnxruntime_prefix
    local enable_gstreamer="OFF"
    local args=(
        -S "${CVKIT_DIR}"
        -B "${CVKIT_DIR}/build/conan/Release"
        -DCMAKE_TOOLCHAIN_FILE="${CVKIT_DIR}/build/conan/Release/generators/conan_toolchain.cmake"
        -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
        -DCMAKE_BUILD_TYPE=Release
        -DCVKIT_BUILD_EXAMPLES="${CVKIT_BUILD_EXAMPLES:-ON}"
        -DCVKIT_BUILD_TESTS="${CVKIT_BUILD_TESTS:-ON}"
        -DCVKIT_BUILD_BENCHMARKS="${CVKIT_BUILD_BENCHMARKS:-OFF}"
        -DCVKIT_ENABLE_ONNXRUNTIME=ON
        -DCVKIT_ENABLE_FFMPEG=OFF
        -DCVKIT_ENABLE_OPENVINO=OFF
        -DCVKIT_ENABLE_TENSORRT=OFF
        -DCVKIT_ENABLE_GSTREAMER_CUDA="${CVKIT_ENABLE_GSTREAMER_CUDA:-OFF}"
        -DCVKIT_CUDART_LIBRARY="${CVKIT_CUDART_LIBRARY:-/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so}"
    )

    opencv_prefix="$(detect_opencv_prefix)"
    onnxruntime_prefix="$(detect_onnxruntime_prefix)"

    if [[ -z "${opencv_prefix}" || -z "${onnxruntime_prefix}" ]]; then
        echo "Skipping cvkit CI: OpenCV or ONNX Runtime prefix not detected"
        exit 0
    fi

    if have_gstreamer; then
        enable_gstreamer="ON"
    fi

    args+=("-DCVKIT_ENABLE_GSTREAMER=${enable_gstreamer}")
    args+=("-DCVKIT_OPENCV_PREFIX=${opencv_prefix}")
    args+=("-DCVKIT_ONNXRUNTIME_PREFIX=${onnxruntime_prefix}")

    if [[ -n "${CVKIT_CUDATOOLKIT_ROOT:-}" ]]; then
        args+=("-DCVKIT_CUDATOOLKIT_ROOT=${CVKIT_CUDATOOLKIT_ROOT}")
    fi

    print_section "cvkit cmake configure"
    cmake "${args[@]}"
}

cmake_build_release() {
    local target="${1:-}"
    print_section "cvkit cmake build"
    if [[ -n "${target}" ]]; then
        cmake --build "${CVKIT_DIR}/build/conan/Release" --target "${target}"
    else
        cmake --build "${CVKIT_DIR}/build/conan/Release"
    fi
}

ctest_release() {
    if [[ "${CVKIT_BUILD_TESTS:-ON}" != "ON" ]]; then
        return
    fi
    print_section "cvkit ctest"
    ctest --test-dir "${CVKIT_DIR}/build/conan/Release" --output-on-failure
}

run_pipeline_image() {
    local binary="${CVKIT_DIR}/build/conan/Release/examples/bin/cvkit_example_pipeline"
    local model="${CVKIT_DIR}/assets/models/yolo11n.onnx"
    local labels="${CVKIT_DIR}/assets/labels/coco80.txt"
    local image="${CVKIT_DIR}/assets/images/test_001.jpg"
    local output_dir="${CVKIT_DIR}/assets/output"

    if [[ ! -x "${binary}" || ! -f "${model}" || ! -f "${labels}" || ! -f "${image}" ]]; then
        echo "Skipping pipeline smoke test: binary or assets missing"
        return
    fi

    print_section "cvkit pipeline image smoke"
    "${binary}" \
        --model "${model}" \
        --labels "${labels}" \
        --image "${image}" \
        --output-dir "${output_dir}"
}

run_media_writer_probe() {
    local script="${CVKIT_DIR}/scripts/probe_media_writer.sh"
    local binary="${CVKIT_DIR}/build/conan/Release/examples/bin/cvkit_example_pipeline"
    local model="${CVKIT_DIR}/assets/models/yolo11n.onnx"
    local labels="${CVKIT_DIR}/assets/labels/coco80.txt"
    local video="${CVKIT_DIR}/assets/video/test.mp4"

    if [[ "${CVKIT_RUN_MEDIA_WRITER_PROBE:-ON}" != "ON" ]]; then
        echo "Skipping media writer probe: CVKIT_RUN_MEDIA_WRITER_PROBE=${CVKIT_RUN_MEDIA_WRITER_PROBE:-}"
        return
    fi

    if [[ ! -x "${script}" || ! -x "${binary}" || ! -f "${model}" || ! -f "${labels}" || ! -f "${video}" ]]; then
        echo "Skipping media writer probe: script, binary, or assets missing"
        return
    fi

    print_section "cvkit media writer probe"
    BUILD_DIR="${CVKIT_DIR}/build/conan/Release" \
        VIDEO_PATH="${video}" \
        MODEL_PATH="${model}" \
        LABELS_PATH="${labels}" \
        OUTPUT_DIR="${CVKIT_DIR}/assets/output" \
        MAX_FRAMES="${CVKIT_MEDIA_WRITER_PROBE_MAX_FRAMES:-2}" \
        RUN_NVENC="${CVKIT_RUN_MEDIA_WRITER_NVENC_PROBE:-0}" \
        "${script}"
}
