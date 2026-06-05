# Status

This file captures the current implementation and validation state. It is a short delivery checklist, not a replacement for `docs/ARCHITECTURE.md`.

## Current Implementation

Implemented public-facing inference scope:

- `detection`
- `face_detection`
- `classification`
- `segmentation`
- `pose`
- `facemesh`
- `promptable_segmentation`

Implemented backend scope:

- ONNX Runtime
- TensorRT

Implemented execution/runtime pieces:

- synchronous `Model::run_sync(...)`
- async `Model::submit(...)`
- graph metadata and graph trace export
- task schemas for all first-stage task families
- public `TaskInput` / `TaskOutput` value contract
- tensor file I/O for EfficientSAM encoder/decoder exchange
- model-level tiled inference through `TileOptions`
- package-mode public API smoke coverage through `test_package`

## Validated Flows

Local CI entrypoint:

- `./scripts/ci/all.sh`

Latest validation result:

- `basekit` package create: passed
- `basekit` test package: passed
- `cvkit` local Conan install/configure/build: passed
- examples build: passed
- CTest on the current ONNX Runtime / CUDA-preprocess Release build: passed, with expected backend-gated skips
- pipeline image smoke: passed
- `cvkit` package create: passed
- `cvkit` test package: `2/2` passed
- TensorRT-enabled Release build: validated on GPU 7
- TensorRT-gated media-to-infer smoke: passed on GPU 7

Model/runtime smoke coverage:

- EfficientSAM encoder-only, decoder-only, combined promptable segmentation
- SCRFD face detection ONNX
- SCRFD tiled face detection on `assets/images/face.jpg`
- OpenCV media `Source` image read and EOF/status behavior
- YOLO11 TensorRT example on GPU 7:
  - `detections=2`
  - graph JSON exported to `assets/output/trt_yolo_graph.json`
- YOLO11 GStreamer/NVDEC -> CUDA preprocess -> TensorRT example on GPU 7:
  - script: `CUDA_DEVICE_INDEX=7 ./scripts/probe_video_gpu_detection.sh`
  - sample result: `frames=3`, `detections=6`, `memory=cuda`, `format=nv12`
  - device convention: physical GPU is selected with `CUDA_VISIBLE_DEVICES`; process-local CUDA device defaults to `0`
  - explicit debug export: `OUTPUT_IMAGE_PATH=/workspace/cvkit/assets/output/video_gpu_detection_frame0.jpg`
  - export sample: `assets/output/video_gpu_detection_frame0.jpg`, `2560x1440`, JPEG
- SCRFD tiled TensorRT example on GPU 7:
  - `faces=657`
  - `keypoints=3285`
- package-mode component targets:
  - `cvkit::core`
  - `cvkit::media`
  - `cvkit::image`
  - `cvkit::infer`
- package-mode aggregate target:
  - `cvkit::cvkit`

Known SCRFD production-compatible path:

- model: `assets/models/scrfd_10g_ac133ba7.onnx`
- family: `scrfd_raw_bgr`
- input: fixed `640x640`, dynamic batch
- preprocessing: direct resize, BGR planar, raw `float32` values in `[0, 255]`, no RGB swap, no normalization
- tiled smoke: `640x640` tiles with `160` overlap
- validated backends: ONNX Runtime and TensorRT

TensorRT cache artifacts currently validated:

- `assets/models/.cvkit_cache/tensorrt/yolo11n.*.plan`
- `assets/models/.cvkit_cache/tensorrt/scrfd_10g_ac133ba7.*.plan`

## Current Constraints

Important limitations:

- Most task pipelines remain host-first.
- Model-level tiling currently requires a host-accessible packed image.
- Promptable segmentation is intentionally not tiled yet.
- Tiled execution records aggregate/per-tile trace entries for both sync `run_sync(...)` and async `submit(...)`, and graph JSON version `6` exposes structured `tiling` / `tile_info` metadata.
- Tiled execution does not yet record full internal task-graph node traces per tile.
- GPU preprocess is still primarily detection-oriented.
- TensorRT smoke tests are opt-in where real GPU/runtime coverage is required.
- Non-float32 execution is not generally supported even though metadata can describe richer dtypes.
- OpenVINO and FFmpeg backend paths are reserved, not implemented.

## Next Recommended Work

Highest-value next steps:

- Continue media runtime cleanup:
  - extend inspectable status beyond the current basic open/eof/error states where practical
  - keep media source lifetime outside the infer task graph
  - keep GStreamer optional and backend-gated
- Add deeper host-first media coverage:
  - keep OpenCV image/video reader tests aligned with workspace assets
  - keep timestamp/fps/source/frame-index metadata coverage
  - add basic writer API symmetry with reader options
- Advance GPU media path:
  - use `scripts/probe_media_gpu.sh` to validate NVDEC on the selected CUDA device
  - use `scripts/probe_video_gpu_detection.sh` to validate NVDEC `DeviceFrame` through TensorRT YOLO detection
  - keep `Source::read(Frame&)` host-only and use `Source::read(DeviceFrame&)` for CUDAMemory output
  - keep CUDAMemory NV12 device frames wired into YOLO CUDA preprocess
  - keep visual export opt-in; do not add implicit full-frame downloads to the GPU path
  - keep TensorRT media-to-infer smoke gated because it requires a real GPU runtime and engine build
- Keep infer follow-up bounded:
  - keep SCRFD raw-BGR compatibility covered
  - add optional full per-tile task-graph node traces only if debugging needs justify the extra verbosity
