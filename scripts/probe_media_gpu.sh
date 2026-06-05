#!/usr/bin/env bash
set -euo pipefail

CVKIT_DIR="${CVKIT_DIR:-/workspace/cvkit}"
VIDEO_PATH="${VIDEO_PATH:-${CVKIT_DIR}/assets/video/test.mp4}"
CUDA_DEVICE_INDEX="${CUDA_DEVICE_INDEX:-7}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"

require_command() {
    local name="$1"
    if ! command -v "${name}" >/dev/null 2>&1; then
        echo "required command not found: ${name}" >&2
        exit 1
    fi
}

has_gst_element() {
    local name="$1"
    gst-inspect-1.0 "${name}" >/dev/null 2>&1
}

print_capability() {
    local name="$1"
    if has_gst_element "${name}"; then
        printf '  %-22s yes\n' "${name}"
    else
        printf '  %-22s no\n' "${name}"
    fi
}

require_command gst-inspect-1.0
require_command gst-launch-1.0
require_command timeout

if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "video file not found: ${VIDEO_PATH}" >&2
    exit 1
fi

device_decoder="nvh264dec"

echo "[gstreamer cuda media capabilities]"
print_capability appsink
print_capability decodebin
print_capability qtdemux
print_capability h264parse
print_capability avdec_h264
print_capability nvh264dec
print_capability "nvh264device${CUDA_DEVICE_INDEX}dec"
print_capability cudaupload
print_capability cudadownload
print_capability cudaconvert

echo
echo "[nvdec cuda-memory smoke]"
echo "video: ${VIDEO_PATH}"
echo "cuda visible device: ${CUDA_DEVICE_INDEX}"
echo "decoder: ${device_decoder}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_INDEX}" timeout "${TIMEOUT_SECONDS}" \
    gst-launch-1.0 -q \
        filesrc "location=${VIDEO_PATH}" ! \
        qtdemux name=demux \
        demux.video_0 ! \
        queue ! \
        h264parse ! \
        "${device_decoder}" ! \
        'video/x-raw(memory:CUDAMemory),format=NV12' ! \
        identity eos-after=8 ! \
        fakesink sync=false async=false

echo "nvdec cuda-memory smoke passed"
