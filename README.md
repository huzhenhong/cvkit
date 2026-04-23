# cvkit

`cvkit` is the vision/media domain library that sits above `basekit`.

Layering:

```text
app
  -> cvkit
    -> basekit
      -> third-party dependencies
```

Current focus:

- YOLO11 ONNX inference
- image/video input with selectable media backends
- annotated image/video output
- clean separation between domain APIs and third-party implementations

## Components

- `cvkit::core`
  - vision-side shared types and stable utility definitions
- `cvkit::media`
  - media backend selection and frame input
- `cvkit::image`
  - image operations used by the inference path
- `cvkit::infer`
  - YOLO11-oriented ONNX Runtime inference pipeline

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
- `compile_commands.json` sync to `build/compile_commands.json`

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
- `--conf`
- `--iou`
- `--max-frames`

TensorRT cache behavior:

- default cache root:
  - `assets/models/.cvkit_cache/tensorrt/`
- default policy:
  - `default`
- cache file naming:
  - model fingerprint + runtime fingerprint
- legacy `*.trt.plan` files are still accepted and migrated to the new cache layout when possible

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

Validated output examples already in the workspace:

- `assets/output/test_001_det.jpg`
- `assets/output/test_det.avi`
- `assets/output/test_det.mp4`

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

Not yet finalized:

- FFmpeg backend
- OpenVINO backend
- production-grade video writer abstraction outside the example support layer
- true asynchronous infer execution beyond the current synchronous `submit()` wrapper

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
  - skips gracefully if required system OpenCV or ONNX Runtime packages are not detected
- `all.sh`
  - runs both scripts in sequence

GitHub Actions example:

- `.github/workflows/ci.yml`

Current CI assumption:

- OpenCV and ONNX Runtime are system-provided dependencies
- GStreamer is optional and auto-detected by the helper scripts
- if these system dependencies are missing, the scripts skip instead of forcing a hard failure
