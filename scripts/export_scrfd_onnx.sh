#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCRFD_TO_ONNX="${SCRFD_TO_ONNX:-/workspace/huzh_work/face/scrfd/tools/scrfd2onnx.py}"
SCRFD_CONFIG="${SCRFD_CONFIG:-/workspace/huzh_work/face/scrfd/configs/scrfd/scrfd_10g_bnkps.py}"
CHECKPOINT="${1:-${ROOT_DIR}/assets/models/scrfd_10g.pth}"
OUTPUT="${2:-${ROOT_DIR}/assets/models/scrfd_10g.onnx}"
INPUT_IMAGE="${SCRFD_EXPORT_INPUT_IMAGE:-${ROOT_DIR}/assets/images/test_001.jpg}"
HEIGHT="${SCRFD_EXPORT_HEIGHT:-640}"
WIDTH="${SCRFD_EXPORT_WIDTH:-640}"

python3 - <<'PY'
try:
    import mmdet  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "mmdet is required to export SCRFD .pth checkpoints. "
        "Run this script from the SCRFD/MMDetection Python environment. "
        f"Import error: {exc}"
    )
PY

python3 "${SCRFD_TO_ONNX}" \
    "${SCRFD_CONFIG}" \
    "${CHECKPOINT}" \
    --shape "${HEIGHT}" "${WIDTH}" \
    --output-file "${OUTPUT}" \
    --input-img "${INPUT_IMAGE}"
