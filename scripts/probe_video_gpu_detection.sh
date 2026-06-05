#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"
BUILD_DIR="${BUILD_DIR:-${CVKIT_DIR}/build/conan/Release}"
CUDA_DEVICE_INDEX="${CUDA_DEVICE_INDEX:-7}"
PROCESS_CUDA_DEVICE_INDEX="${PROCESS_CUDA_DEVICE_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-3}"

EXAMPLE_BIN="${BUILD_DIR}/examples/bin/cvkit_example_video_gpu_detection"
VIDEO_PATH="${VIDEO_PATH:-${CVKIT_DIR}/assets/video/test.mp4}"
MODEL_PATH="${MODEL_PATH:-${CVKIT_DIR}/assets/models/yolo11n.onnx}"
LABELS_PATH="${LABELS_PATH:-${CVKIT_DIR}/assets/labels/coco80.txt}"
OUTPUT_IMAGE_PATH="${OUTPUT_IMAGE_PATH:-}"
DUMP_FRAME_INDEX="${DUMP_FRAME_INDEX:-0}"

if [[ ! -x "${EXAMPLE_BIN}" ]]; then
    echo "example binary not found or not executable: ${EXAMPLE_BIN}" >&2
    echo "build it with CVKIT_ENABLE_TENSORRT=ON and CVKIT_BUILD_EXAMPLES=ON" >&2
    exit 1
fi

echo "cuda visible device: ${CUDA_DEVICE_INDEX}"
echo "process cuda device: ${PROCESS_CUDA_DEVICE_INDEX}"
echo "video: ${VIDEO_PATH}"
echo "model: ${MODEL_PATH}"
echo "max frames: ${MAX_FRAMES}"
if [[ -n "${OUTPUT_IMAGE_PATH}" ]]; then
    echo "output image: ${OUTPUT_IMAGE_PATH}"
    echo "dump frame: ${DUMP_FRAME_INDEX}"
fi

args=(
    --video "${VIDEO_PATH}" \
    --model "${MODEL_PATH}" \
    --labels "${LABELS_PATH}" \
    --cuda-device "${PROCESS_CUDA_DEVICE_INDEX}" \
    --max-frames "${MAX_FRAMES}"
)

if [[ -n "${OUTPUT_IMAGE_PATH}" ]]; then
    args+=(--output-image "${OUTPUT_IMAGE_PATH}" --dump-frame "${DUMP_FRAME_INDEX}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_INDEX}" "${EXAMPLE_BIN}" "${args[@]}"
