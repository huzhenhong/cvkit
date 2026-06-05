# cvkit

`cvkit` is the vision/media domain library that sits above `basekit`.

Architecture details:

- `docs/ARCHITECTURE.md`
- `docs/CODING_STYLE.md`

Layering:

```text
app
  -> cvkit
    -> basekit
      -> third-party dependencies
```

Current focus:

- YOLO11 detection with ONNX Runtime and TensorRT
- EfficientSAM-style promptable segmentation with encoder/decoder split
- image/video input with selectable media backends
- graph-aware async execution, tracing, and JSON export
- clean separation between domain APIs and third-party implementations

## Components

- `cvkit::core`
  - vision-side shared types and stable utility definitions
- `cvkit::media`
  - media backend selection and frame input
- `cvkit::image`
  - image operations used by the inference path
- `cvkit::infer`
  - multi-backend inference, task graph execution, and task-oriented pipelines

## Third-Party Strategy

`cvkit` consumes:

- `basekit` through Conan
- OpenCV from the local system via `find_package(OpenCV ...)`
- ONNX Runtime from the local system via `find_package(onnxruntime CONFIG REQUIRED)`
- GStreamer from the local system via `PkgConfig`

Consumer rule for passthrough dependencies:

- when `cvkit` needs passthrough third-party APIs exposed by `basekit`, prefer `#include <basekit/ext/bk_*.h>`
- do not directly include passthrough headers such as `CLI11`, `fmt`, or `Tracy` in normal `cvkit` code unless there is a deliberate exception

Reserved but not implemented yet:

- FFmpeg backend
- OpenVINO backend

Current first-stage inference scope:

- implemented tasks:
  - detection
  - face detection
  - classification
  - segmentation
  - pose
  - facemesh
  - promptable segmentation
- implemented backends:
  - onnxruntime
  - tensorrt
## Build Options

Core local options:

- `CVKIT_BUILD_EXAMPLES`
- `CVKIT_BUILD_TESTS`
- `CVKIT_BUILD_BENCHMARKS`

Media/inference backend options:

- `CVKIT_ENABLE_GSTREAMER`
- `CVKIT_ENABLE_GSTREAMER_CUDA`
- `CVKIT_ENABLE_FFMPEG`
- `CVKIT_ENABLE_ONNXRUNTIME`
- `CVKIT_ENABLE_OPENVINO`
- `CVKIT_ENABLE_TENSORRT`

Optional dependency path hints:

- `CVKIT_OPENCV_PREFIX`
- `CVKIT_ONNXRUNTIME_PREFIX`
- `CVKIT_FFMPEG_PREFIX`
- `CVKIT_OPENVINO_PREFIX`
- `CVKIT_TENSORRT_PREFIX`
- `CVKIT_CUDATOOLKIT_ROOT`
- `CVKIT_CUDART_LIBRARY`

`CMAKE_PREFIX_PATH` remains the main search mechanism. These cache variables only prepend extra lookup locations when needed.

## Assets Layout

Current workspace assets:

```text
/workspace/cvkit/assets
├── images/
├── video/
├── labels/
│   └── coco80.txt
├── models/
│   ├── README.md
│   └── yolo11n.onnx
└── output/
```

Conventions:

- labels file format: UTF-8 text, one class name per line
- default labels: `assets/labels/coco80.txt`
- default model path: `assets/models/yolo11n.onnx`
- generated example outputs go to `assets/output/`

Current promptable-segmentation assets:

- encoder model:
  - `assets/models/efficient_sam_vitt_encoder.sim.onnx`
- decoder model:
  - `assets/models/efficient_sam_vitt_decoder.sim.onnx`

## Local Build

Package and toolchain preparation:

```bash
cd /workspace/cvkit
/root/.local/bin/conan install . -of build -s compiler.cppstd=17 --build=missing
```

Configure and build:

```bash
cmake -S . -B build/conan/Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/Release/generators/conan_toolchain.cmake \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVKIT_BUILD_EXAMPLES=ON \
  -DCVKIT_BUILD_TESTS=ON \
  -DCVKIT_BUILD_BENCHMARKS=ON
cmake --build build/conan/Release
ctest --test-dir build/conan/Release --output-on-failure
```

When a consumed `basekit` package changes, rerun both:

```bash
/root/.local/bin/conan install . -of build -s compiler.cppstd=17 --build=missing
cmake -S . -B build/conan/Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/Release/generators/conan_toolchain.cmake \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release
```

This refreshes Conan-generated CMake metadata, including the `basekitExtras.cmake` bridge used for packaged passthrough targets such as `basekit::cli11`.

## Local Scripts

Main local entrypoints:

- `./scripts/dev_build.sh`
- `./scripts/dev_build.sh debug`
- `./scripts/dev_build.sh both`
- `./scripts/clean.sh`

`dev_build.sh` wraps:

- `conan install`
- `cmake configure`
- `cmake build`
- `ctest`
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `compile_commands.json` sync to `build/compile_commands.json`

Current script behavior notes:

- default mode is `release`
- after a successful build, the active build tree's `compile_commands.json` is copied to:
  - `build/compile_commands.json`
- TensorRT-specific smoke tests remain opt-in and still require:
  - `CVKIT_RUN_TENSORRT_SMOKE=1`

Useful environment overrides:

- `BUILD_EXAMPLES=ON|OFF`
- `BUILD_TESTS=ON|OFF`
- `BUILD_BENCHMARKS=ON|OFF`
- `ENABLE_GSTREAMER=ON|OFF`
- `ENABLE_GSTREAMER_CUDA=ON|OFF`
- `ENABLE_FFMPEG=ON|OFF`
- `ENABLE_ONNXRUNTIME=ON|OFF`
- `ENABLE_OPENVINO=ON|OFF`
- `ENABLE_TENSORRT=ON|OFF`
- `OPENCV_PREFIX`
- `ONNXRUNTIME_PREFIX`
- `CUDATOOLKIT_ROOT`
- `CUDART_LIBRARY`
- `CVKIT_EXECUTOR_THREADS`
- `CVKIT_TRACE_GRAPH`

Example:

```bash
cd /workspace/cvkit
ENABLE_GSTREAMER_CUDA=ON BUILD_BENCHMARKS=OFF ./scripts/dev_build.sh
```

## Pipeline Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_pipeline`

Supported CLI options:

- `--model`
- `--labels`
- `--image`
- `--video`
- `--output-dir`
- `--reader opencv|gstreamer|ffmpeg`
- `--writer opencv|gstreamer|ffmpeg`
- `--infer-backend onnxruntime|tensorrt`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--gst-codec jpegavi|x264mp4|nvh264|nvv4l2h264`
- `--async`
- `--print-graph`
- `--dump-graph-json`
- `--conf`
- `--iou`
- `--max-frames`

## Classification Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_classification`

Supported CLI options:

- `--model`
- `--labels`
- `--image`
- `--infer-backend onnxruntime|tensorrt`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--async`
- `--print-graph`
- `--dump-graph-json`

Example:

```bash
cd /workspace/cvkit
build/conan/Release/examples/bin/cvkit_example_classification \
  --model /path/to/classification.onnx \
  --labels /path/to/labels.txt \
  --image assets/images/test_001.jpg \
  --infer-backend onnxruntime \
  --print-graph
```

## Face Detection Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_face_detection`

Supported CLI options:

- `--model`
- `--image`
- `--output-dir`
- `--infer-backend onnxruntime|tensorrt`
- `--family scrfd|scrfd_raw_bgr`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--async`
- `--print-graph`
- `--dump-graph-json`
- `--conf`
- `--iou`
- `--tile-width`
- `--tile-height`
- `--tile-overlap-x`
- `--tile-overlap-y`

Example:

```bash
cd /workspace/cvkit
build/conan/Release/examples/bin/cvkit_example_face_detection \
  --model assets/models/scrfd_10g_ac133ba7.onnx \
  --image assets/images/face.jpg \
  --output-dir assets/output \
  --infer-backend onnxruntime \
  --family scrfd_raw_bgr \
  --conf 0.5 \
  --iou 0.4 \
  --tile-width 640 \
  --tile-height 640 \
  --tile-overlap-x 160 \
  --tile-overlap-y 160 \
  --print-graph
```

`scrfd_raw_bgr` keeps the production SCRFD preprocessing contract: direct resize, BGR planar input, raw `float32` values in `[0, 255]`, and no RGB swap or normalization. The tile flags enable model-level tiled inference and merge detections back into source-image coordinates.

The bundled `assets/models/scrfd_10g.pth` is a PyTorch/MMDetection checkpoint, not an ONNX Runtime or TensorRT loadable model. Export it first from an environment that has SCRFD/MMDetection dependencies:

```bash
cd /workspace/cvkit
scripts/export_scrfd_onnx.sh assets/models/scrfd_10g.pth assets/models/scrfd_10g.onnx
```

## Segmentation Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_segmentation`

Supported CLI options:

- `--model`
- `--image`
- `--output-dir`
- `--infer-backend onnxruntime|tensorrt`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--async`
- `--print-graph`
- `--dump-graph-json`

Example:

```bash
cd /workspace/cvkit
build/conan/Release/examples/bin/cvkit_example_segmentation \
  --model /path/to/segmentation.onnx \
  --image assets/images/test_001.jpg \
  --output-dir assets/output \
  --infer-backend onnxruntime \
  --print-graph
```

## Pose Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_pose`

Supported CLI options:

- `--model`
- `--image`
- `--infer-backend onnxruntime|tensorrt`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--async`
- `--print-graph`
- `--dump-graph-json`

Example:

```bash
cd /workspace/cvkit
build/conan/Release/examples/bin/cvkit_example_pose \
  --model /path/to/pose.onnx \
  --image assets/images/test_001.jpg \
  --infer-backend onnxruntime \
  --print-graph
```

## FaceMesh Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_facemesh`

Supported CLI options:

- `--model`
- `--image`
- `--infer-backend onnxruntime|tensorrt`
- `--cache-policy default|disabled|read-only|rebuild`
- `--cache-dir`
- `--async`
- `--print-graph`
- `--dump-graph-json`

Example:

```bash
cd /workspace/cvkit
build/conan/Release/examples/bin/cvkit_example_facemesh \
  --model /path/to/facemesh.onnx \
  --image assets/images/test_001.jpg \
  --infer-backend onnxruntime \
  --print-graph
```

TensorRT cache behavior:

- default cache root:
  - `assets/models/.cvkit_cache/tensorrt/`
- default policy:
  - `default`
- cache file naming:
  - model fingerprint + runtime fingerprint
- legacy `*.trt.plan` files are still accepted and migrated to the new cache layout when possible
- `ModelSpec::tensorrt_profiles`
  - optional per-input override for TensorRT optimization profile shapes
  - lets you provide explicit `min/opt/max` shapes by input tensor name
  - if omitted, `cvkit` falls back to task-aware defaults for the current detection and promptable-segmentation paths
- `ModelSpec::tensorrt_prefer_device_outputs`
  - optional TensorRT-only hint to keep backend output metadata and raw outputs CUDA-resident when possible
  - current detection path can still run with this enabled because detection postprocess materializes host copies when needed

Current public data-contract notes:

- `TaskInput` / `TaskOutput`
  - support richer value types instead of only raw `Frame` and detection lists
- `ImageValue`
  - carries:
    - `frame`
    - `memory_device`
    - `device`
    - `storage`
    - `row_stride_bytes`
    - `external_data`
    - `storage_bytes`
  - current examples and host-side pipelines use `StorageKind::owned`
  - host processing currently requires a valid host layout
  - detection now also accepts `memory_device=cuda` image inputs when a valid external device view is provided
- `TensorValue`
  - carries:
    - `name`
    - `shape`
    - `data`
    - `data_type`
    - `memory_device`
    - `storage`
    - `external_data`
    - `storage_bytes`
  - current generic execution contract accepts host `float32` input tensors
  - current TensorRT path additionally supports CUDA-resident external-view `float32` input tensors
- classification first-stage status:
  - current pipeline is host-first and task-oriented
  - it resizes and normalizes image input on CPU
  - it returns:
    - `classification`
    - `scores`
  - example entrypoint is now available:
    - `build/conan/Release/examples/bin/cvkit_example_classification`
  - there is no bundled classification model asset yet, so repository coverage is currently provided by focused unit tests and stub backends
- segmentation first-stage status:
  - current pipeline is host-first and task-oriented
  - it resizes and normalizes image input on CPU
  - it treats backend output as `NCHW` class logits
  - it returns:
    - `mask`
    - `logits`
  - example entrypoint is now available:
    - `build/conan/Release/examples/bin/cvkit_example_segmentation`
  - there is no bundled segmentation model asset yet, so repository coverage is currently provided by focused unit tests and stub backends
- face detection first-stage status:
  - current pipeline is host-first and task-oriented
  - it is intended for SCRFD-style outputs that combine bbox, score, and sparse facial keypoints
  - it supports SCRFD 9-output ONNX exports:
    - `score_8`, `score_16`, `score_32`
    - `bbox_8`, `bbox_16`, `bbox_32`
    - `kps_8`, `kps_16`, `kps_32`
  - it returns:
    - `detections`
  - each `Detection` may now carry `keypoints`, so a face can expose bbox + score + 5 landmarks in one result item
  - example entrypoint is now available:
    - `build/conan/Release/examples/bin/cvkit_example_face_detection`
  - there is no bundled face detection model asset yet, so repository coverage is currently provided by focused unit tests and stub backends
- pose first-stage status:
  - current pipeline is host-first and task-oriented
  - it resizes and normalizes image input on CPU
  - it interprets the first backend output as packed keypoint coordinates with optional score values
  - it returns:
    - `keypoints`
    - `scores`
    - `raw`
  - example entrypoint is now available:
    - `build/conan/Release/examples/bin/cvkit_example_pose`
  - there is no bundled pose model asset yet, so repository coverage is currently provided by focused unit tests and stub backends
- facemesh first-stage status:
  - current pipeline is host-first and task-oriented
  - it resizes and normalizes image input on CPU
  - it interprets the first backend output as packed landmark coordinates with optional score values
  - it returns:
    - `landmarks`
    - `scores`
    - `raw`
  - example entrypoint is now available:
    - `build/conan/Release/examples/bin/cvkit_example_facemesh`
  - there is no bundled facemesh model asset yet, so repository coverage is currently provided by focused unit tests and stub backends
- promptable segmentation decoder can now consume CUDA-resident `image_embeddings` by materializing them back to host before the current ONNX Runtime execution path
- promptable segmentation encoder and combined flows can now also accept CUDA-resident `ImageValue` inputs; current ONNX Runtime execution still materializes them back to host before encoder execution
- when an ONNX Runtime session is loaded with `ModelSpec.device.kind = cuda`, CUDA-resident promptable decoder embeddings can now flow through a real ORT CUDA tensor input path instead of always being copied back to host first
  - current backend export contract still emits `float32` tensors only
- tensor file format:
  - current writer emits `CVKTEMB3`
  - readers remain backward-compatible with `CVKTEMB2` and `CVKTEMB1`

Current detection CUDA-preprocess status:

- if CUDA language support is available at configure time, `cvkit::infer` builds a device-side YOLO preprocess kernel
- if only `cudart` is available, detection still accepts `ImageValue(memory_device=cuda)` but currently falls back to:
  - device-to-host copy
  - existing CPU preprocess
- on the TensorRT path with CUDA language enabled, detection can now produce a CUDA-resident float32 input tensor and bind it directly into TensorRT without an extra host-to-device upload
- current CMake autodiscovery prefers:
  - `/usr/local/cuda-13.0/bin/nvcc`
  - then `/usr/local/bin/nvcc`

Execution and graph tracing:

- `--async`
  - runs inference through `Model::submit()` instead of the direct synchronous path
- `--print-graph`
  - prints the task graph nodes, inferred graph boundary, and the most recent per-node trace from the example process
  - useful when checking `consumes` / `produces` contracts and the actual node order without attaching a debugger
- `--dump-graph-json`
  - writes the same graph metadata and latest trace to a JSON file
  - useful for scripting, visualization, and offline DAG inspection
  - current schema version: `5`
- `CVKIT_EXECUTOR_THREADS`
  - controls the shared internal executor thread-pool size
- `CVKIT_TRACE_GRAPH=1`
  - enables per-node graph timing logs through `basekit::log`
  - each node log includes:
    - node name
    - sequence index
    - duration in microseconds
    - input count
    - output delta
    - scratch delta

Example `--print-graph` output:

```text
graph.nodes=2
  node=detection_infer consumes=input:image produces=scratch:detection.preprocess,scratch:detection.raw_outputs
  node=detection_postprocess depends_on=detection_infer consumes=input:image,scratch:detection.preprocess,scratch:detection.raw_outputs produces=output:detections
graph.inputs=input:image
graph.outputs=output:detections
graph.trace.nodes=2
  trace.node=detection_infer seq=0 duration_us=114473 input_count=1 output_count=0 scratch_count=2
  trace.node=detection_postprocess seq=1 duration_us=2741 input_count=1 output_count=1 scratch_count=0
```

Example JSON dump:

```bash
CUDA_VISIBLE_DEVICES=7 build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend tensorrt \
  --image assets/images/test_001.jpg \
  --output-dir assets/output \
  --async \
  --dump-graph-json assets/output/graph.json
```

Image run:

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend onnxruntime \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --output-dir /workspace/cvkit/assets/output
```

Video run with GStreamer H.264 MP4 output:

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --reader gstreamer \
  --writer gstreamer \
  --infer-backend onnxruntime \
  --gst-codec x264mp4 \
  --video /workspace/cvkit/assets/video/test.mp4 \
  --output-dir /workspace/cvkit/assets/output \
  --max-frames 30
```

TensorRT run with explicit cache control:

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend tensorrt \
  --cache-policy rebuild \
  --cache-dir /workspace/cvkit/assets/cache/trt \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --model /workspace/cvkit/assets/models/yolo11n.onnx \
  --labels /workspace/cvkit/assets/labels/coco80.txt \
  --output-dir /workspace/cvkit/assets/output
```

TensorRT run with async infer and graph timing:

```bash
CVKIT_TRACE_GRAPH=1 CUDA_VISIBLE_DEVICES=7 \
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend tensorrt \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --output-dir /workspace/cvkit/assets/output \
  --async
```

Validated output examples already in the workspace:

- `assets/output/test_001_det.jpg`
- `assets/output/test_det.avi`
- `assets/output/test_det.mp4`

## Promptable Segmentation Example

Example binary:

- `build/conan/Release/examples/bin/cvkit_example_promptable_segmentation`

Current first-stage implementation:

- backend:
  - `onnxruntime`
- family:
  - `efficient_sam`
  - `efficient_sam_encoder`
  - `efficient_sam_decoder`
- model layout:
  - `--encoder` points to `efficient_sam_vitt_encoder.sim.onnx`
  - `--decoder` points to `efficient_sam_vitt_decoder.sim.onnx`
- run modes:
  - `--mode combined`
    - runs encoder + decoder end-to-end
  - `--mode encoder`
    - runs only the encoder and writes `image_embeddings`
  - `--mode decoder`
    - consumes a previously exported `image_embeddings` file and runs only the decoder
- supported prompts:
  - single or multiple point prompts through `--point-x/--point-y`
  - optional box prompt through `--use-box --box-x --box-y --box-w --box-h`
- embedding exchange:
  - use `--embeddings <path>`
  - current example format is a compact cvkit binary tensor file (`*.bin`)

Example run:

```bash
/tmp/cvkit-ort-out/conan/Release/examples/bin/cvkit_example_promptable_segmentation \
  --mode combined \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --point-x 640 \
  --point-y 360 \
  --output-dir /workspace/cvkit/assets/output \
  --print-graph \
  --dump-graph-json /workspace/cvkit/assets/output/efficient_sam_graph.json
```

Encoder-only export:

```bash
/tmp/cvkit-ort-out/conan/Release/examples/bin/cvkit_example_promptable_segmentation \
  --mode encoder \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --point-x 640 \
  --point-y 360 \
  --embeddings /workspace/cvkit/assets/output/test_001_sam_embeddings.bin
```

Decoder-only run:

```bash
/tmp/cvkit-ort-out/conan/Release/examples/bin/cvkit_example_promptable_segmentation \
  --mode decoder \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --point-x 640 \
  --point-y 360 \
  --embeddings /workspace/cvkit/assets/output/test_001_sam_embeddings.bin \
  --output-dir /workspace/cvkit/assets/output
```

Validated outputs now present in the workspace:

- `assets/output/test_001_sam_mask.png`
- `assets/output/test_001_sam_overlay.png`
- `assets/output/efficient_sam_graph.json`
- `assets/output/test_001_sam_embeddings.bin`

## Current State

Verified locally:

- Conan install works
- `cvkit` configures and builds
- local tests pass
- YOLO11 image inference path works
- GStreamer video read path works
- GStreamer `jpegavi` and `x264mp4` write paths work
- `CVKIT_ENABLE_GSTREAMER_CUDA=ON` configure/build path has been validated
- TensorRT backend load/run/cache path has been validated on GPU 7
- TensorRT serialized engine cache supports fingerprinted cache files and legacy cache migration
- internal async executor path has been validated
- graph-aware async execution is active for detection and promptable segmentation
- task graph metadata and per-node timing trace are available
- `cvkit::infer::Model::session_info()` now exposes backend tensor input/output metadata through the public API
- `ImageValue` and `TensorValue` now carry minimal device-aware contract metadata
- examples now use `ImageValue` as the primary image input object
- tensor session/debug metadata now includes:
  - `data_type`
  - `memory_device`
- tensor/image contracts now distinguish:
  - host layout validity
  - storage kind (`owned` vs `external_view`)

Current note on local backend selection:

- some local build trees may have `CVKIT_ENABLE_ONNXRUNTIME=OFF`
- in those trees, the example should be run with `--infer-backend tensorrt` instead of relying on the default

Not yet finalized:

- FFmpeg backend
- OpenVINO backend
- production-grade video writer abstraction outside the example support layer
- generalized non-host / non-float32 tensor execution paths
- production GPU preprocess and broader device-resident data flow
- more task families beyond detection, classification, segmentation, and promptable segmentation
- stronger external-view / zero-copy ownership model beyond the current metadata-only `storage` contract

## Planned Next

Near-term work planned from the current codebase state:

- continue device-aware data-path work
  - keep public contracts stable while preparing for CUDA-backed image/tensor paths
- move from metadata-only device awareness toward real GPU preprocess support
- continue tightening the backend tensor-engine contract
  - especially around non-host and non-float32 execution support
- add model-family adapters for pose and facemesh once concrete model assets are selected

Important scope note:

- these are planned directions, not promises of current production readiness
- the current repository state is strongest in:
  - YOLO11 detection
  - EfficientSAM promptable segmentation
  - task-graph execution and observability

## CI Entrypoints

Available CI helper scripts:

- `scripts/ci/cvkit_local.sh`
- `scripts/ci/cvkit_package.sh`
- `scripts/ci/all.sh`

Behavior:

- `cvkit_local.sh`
  - optionally creates a fresh `basekit` package first
  - installs `cvkit`
  - configures and builds the Release tree
  - runs local tests
  - runs an image pipeline smoke test if the required assets are present
- `cvkit_package.sh`
  - runs `conan create` for `cvkit`
  - builds and runs `test_package`, which validates packaged public headers, component targets, the aggregate `cvkit::cvkit` target, task schemas, and `TileOptions` without requiring model assets
  - skips gracefully if required system OpenCV or ONNX Runtime packages are not detected
- `all.sh`
  - runs both scripts in sequence

GitHub Actions example:

- `.github/workflows/ci.yml`

Current CI assumption:

- OpenCV and ONNX Runtime are system-provided dependencies
- GStreamer is optional and auto-detected by the helper scripts
- if these system dependencies are missing, the scripts skip instead of forcing a hard failure
