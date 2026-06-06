#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"
BUILD_DIR="${BUILD_DIR:-${CVKIT_DIR}/build/conan/Release}"
VIDEO_PATH="${VIDEO_PATH:-${CVKIT_DIR}/assets/video/test.mp4}"
MODEL_PATH="${MODEL_PATH:-${CVKIT_DIR}/assets/models/yolo11n.onnx}"
LABELS_PATH="${LABELS_PATH:-${CVKIT_DIR}/assets/labels/coco80.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${CVKIT_DIR}/assets/output}"
MAX_FRAMES="${MAX_FRAMES:-2}"
INFER_BACKEND="${INFER_BACKEND:-onnxruntime}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
RUN_NVENC="${RUN_NVENC:-0}"

PIPELINE_BIN="${BUILD_DIR}/examples/bin/cvkit_example_pipeline"
MEDIA_TEST_BIN="${BUILD_DIR}/tests/bin/cvkit_test_media"

require_command() {
    local name="$1"
    if ! command -v "${name}" >/dev/null 2>&1; then
        echo "required command not found: ${name}" >&2
        exit 1
    fi
}

has_gst_element() {
    local name="$1"
    command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 "${name}" >/dev/null 2>&1
}

print_gst_element() {
    local name="$1"
    if has_gst_element "${name}"; then
        printf '  %-14s yes\n' "${name}"
    else
        printf '  %-14s no\n' "${name}"
    fi
}

has_gst_writer_base() {
    has_gst_element appsrc && has_gst_element videoconvert
}

has_gst_jpegavi_writer() {
    has_gst_writer_base && has_gst_element jpegenc && has_gst_element avimux
}

has_gst_x264mp4_writer() {
    has_gst_writer_base && has_gst_element x264enc && has_gst_element h264parse && has_gst_element mp4mux
}

has_gst_nvh264_writer() {
    has_gst_writer_base && has_gst_element nvh264enc && has_gst_element h264parse && has_gst_element mp4mux
}

run_pipeline_writer() {
    local reader="$1"
    local writer="$2"
    local codec="$3"
    local label="$4"

    echo
    echo "[${label}]"
    timeout "${TIMEOUT_SECONDS}" "${PIPELINE_BIN}" \
        --video "${VIDEO_PATH}" \
        --model "${MODEL_PATH}" \
        --labels "${LABELS_PATH}" \
        --max-frames "${MAX_FRAMES}" \
        --infer-backend "${INFER_BACKEND}" \
        --reader "${reader}" \
        --writer "${writer}" \
        --gst-codec "${codec}" \
        --output-dir "${OUTPUT_DIR}"
}

require_command timeout

if [[ ! -x "${PIPELINE_BIN}" ]]; then
    echo "pipeline example not found or not executable: ${PIPELINE_BIN}" >&2
    echo "build it with CVKIT_BUILD_EXAMPLES=ON" >&2
    exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "video file not found: ${VIDEO_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "model file not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${LABELS_PATH}" ]]; then
    echo "labels file not found: ${LABELS_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[media writer probe]"
echo "build: ${BUILD_DIR}"
echo "video: ${VIDEO_PATH}"
echo "model: ${MODEL_PATH}"
echo "max frames: ${MAX_FRAMES}"
echo "infer backend: ${INFER_BACKEND}"
echo "output dir: ${OUTPUT_DIR}"

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
    echo
    echo "[gstreamer writer elements]"
    print_gst_element appsrc
    print_gst_element videoconvert
    print_gst_element jpegenc
    print_gst_element avimux
    print_gst_element x264enc
    print_gst_element h264parse
    print_gst_element mp4mux
    print_gst_element nvh264enc
fi

if [[ -x "${MEDIA_TEST_BIN}" ]]; then
    echo
    echo "[writer unit smoke]"
    timeout "${TIMEOUT_SECONDS}" "${MEDIA_TEST_BIN}" --reporter compact
fi

run_pipeline_writer opencv opencv jpegavi "pipeline opencv reader -> opencv writer"

if has_gst_jpegavi_writer; then
    run_pipeline_writer opencv gstreamer jpegavi "pipeline opencv reader -> gstreamer jpegavi writer"
else
    echo
    echo "[pipeline opencv reader -> gstreamer jpegavi writer]"
    echo "skipped; required GStreamer jpegavi writer elements are not available"
fi

if has_gst_x264mp4_writer; then
    run_pipeline_writer opencv gstreamer x264mp4 "pipeline opencv reader -> gstreamer x264mp4 writer"
else
    echo
    echo "[pipeline opencv reader -> gstreamer x264mp4 writer]"
    echo "skipped; required GStreamer x264mp4 writer elements are not available"
fi

if [[ "${RUN_NVENC}" == "1" ]]; then
    if has_gst_nvh264_writer; then
        run_pipeline_writer opencv gstreamer nvh264 "pipeline opencv reader -> gstreamer nvh264 writer"
    else
        echo
        echo "[pipeline opencv reader -> gstreamer nvh264 writer]"
        echo "skipped; required GStreamer nvh264 writer elements are not available"
    fi
else
    echo
    echo "[pipeline opencv reader -> gstreamer nvh264 writer]"
    echo "skipped; set RUN_NVENC=1 to run the environment-dependent NVENC writer probe"
fi

echo
echo "media writer probe completed"
