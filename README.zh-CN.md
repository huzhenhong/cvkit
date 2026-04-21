# cvkit

`cvkit` 是位于 `basekit` 之上的视觉/媒体领域库。

分层关系：

```text
app
  -> cvkit
    -> basekit
      -> third-party dependencies
```

当前阶段重点：

- YOLO11 ONNX 推理
- 可切换后端的图像/视频输入
- 带检测框绘制的图像/视频输出
- 保持领域 API 与第三方实现之间的边界清晰

## 组件

- `cvkit::core`
  - 视觉侧共享类型与稳定基础定义
- `cvkit::media`
  - 媒体后端选择与帧输入
- `cvkit::image`
  - 推理流程所需的图像处理
- `cvkit::infer`
  - 面向 YOLO11 的 ONNX Runtime 推理流程

## 第三方依赖策略

`cvkit` 当前依赖：

- 通过 Conan 消费 `basekit`
- 通过 `find_package(OpenCV ...)` 使用本机 OpenCV
- 通过 `find_package(onnxruntime CONFIG REQUIRED)` 使用本机 ONNX Runtime
- 通过 `PkgConfig` 使用本机 GStreamer

对透传依赖的 consumer 规则：

- 当 `cvkit` 需要使用 `basekit` 暴露出来的第三方透传 API 时，优先包含 `#include <basekit/ext/bk_*.h>`
- 除非有明确例外，不要在普通 `cvkit` 代码里直接包含 `CLI11`、`fmt`、`Tracy` 这类透传头

已经预留但暂未实现：

- FFmpeg backend
- OpenVINO backend
- TensorRT backend

## 构建开关

本地目标开关：

- `CVKIT_BUILD_EXAMPLES`
- `CVKIT_BUILD_TESTS`
- `CVKIT_BUILD_BENCHMARKS`

媒体/推理后端开关：

- `CVKIT_ENABLE_GSTREAMER`
- `CVKIT_ENABLE_GSTREAMER_CUDA`
- `CVKIT_ENABLE_FFMPEG`
- `CVKIT_ENABLE_ONNXRUNTIME`
- `CVKIT_ENABLE_OPENVINO`
- `CVKIT_ENABLE_TENSORRT`

可选依赖路径提示：

- `CVKIT_OPENCV_PREFIX`
- `CVKIT_ONNXRUNTIME_PREFIX`
- `CVKIT_FFMPEG_PREFIX`
- `CVKIT_OPENVINO_PREFIX`
- `CVKIT_TENSORRT_PREFIX`
- `CVKIT_CUDATOOLKIT_ROOT`
- `CVKIT_CUDART_LIBRARY`

默认仍然以 `CMAKE_PREFIX_PATH` 为主，上述 cache 变量只是用于在需要时向查找路径前置额外位置。

## 资源目录约定

当前工作区资源布局：

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

约定：

- 标签文件格式：UTF-8 文本，一行一个类别
- 默认标签文件：`assets/labels/coco80.txt`
- 默认模型路径：`assets/models/yolo11n.onnx`
- example 生成的输出默认写到 `assets/output/`

## 本地构建

先准备 Conan 依赖和 toolchain：

```bash
cd /workspace/cvkit
/root/.local/bin/conan install . -of build -s compiler.cppstd=17 --build=missing
```

再 configure / build：

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

如果消费的 `basekit` package 有更新，需要重新执行：

```bash
/root/.local/bin/conan install . -of build -s compiler.cppstd=17 --build=missing
cmake -S . -B build/conan/Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/Release/generators/conan_toolchain.cmake \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release
```

这是为了刷新 Conan 生成的 CMake 元数据，包括 `basekitExtras.cmake` 提供的 package 态透传 target 桥接能力，例如 `basekit::cli11`。

## 本地脚本

当前本地入口脚本：

- `./scripts/dev_build.sh`
- `./scripts/dev_build.sh debug`
- `./scripts/dev_build.sh both`
- `./scripts/clean.sh`

`dev_build.sh` 封装了：

- `conan install`
- `cmake configure`
- `cmake build`
- `ctest`
- 把当前配置的 `compile_commands.json` 同步到 `build/compile_commands.json`

常用环境变量覆盖项：

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

示例：

```bash
cd /workspace/cvkit
ENABLE_GSTREAMER_CUDA=ON BUILD_BENCHMARKS=OFF ./scripts/dev_build.sh
```

## Pipeline Example

示例可执行文件：

- `build/conan/Release/examples/bin/cvkit_example_pipeline`

当前支持的 CLI 参数：

- `--model`
- `--labels`
- `--image`
- `--video`
- `--output-dir`
- `--reader opencv|gstreamer|ffmpeg`
- `--writer opencv|gstreamer|ffmpeg`
- `--gst-codec jpegavi|x264mp4|nvh264|nvv4l2h264`
- `--conf`
- `--iou`
- `--max-frames`

图片推理示例：

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --output-dir /workspace/cvkit/assets/output
```

GStreamer 视频读写加 H.264 MP4 输出示例：

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --reader gstreamer \
  --writer gstreamer \
  --gst-codec x264mp4 \
  --video /workspace/cvkit/assets/video/test.mp4 \
  --output-dir /workspace/cvkit/assets/output \
  --max-frames 30
```

当前工作区里已经生成过的验证输出：

- `assets/output/test_001_det.jpg`
- `assets/output/test_det.avi`
- `assets/output/test_det.mp4`

## 当前状态

已在本地验证：

- Conan install 正常
- `cvkit` configure / build 正常
- 本地 tests 通过
- YOLO11 图片推理链路正常
- GStreamer 视频读取正常
- GStreamer `jpegavi` 和 `x264mp4` 写出链路正常
- `CVKIT_ENABLE_GSTREAMER_CUDA=ON` 的 configure / build 路径已经验证

尚未定稿：

- FFmpeg backend
- OpenVINO backend
- TensorRT backend
- 从 example 支持层抽离成正式生产级 writer 抽象

## CI 入口

当前提供的 CI 辅助脚本：

- `scripts/ci/cvkit_local.sh`
- `scripts/ci/cvkit_package.sh`
- `scripts/ci/all.sh`

行为说明：

- `cvkit_local.sh`
  - 可选地先创建一份新的 `basekit` package
  - 安装 `cvkit`
  - configure / build Release 构建树
  - 运行本地 tests
  - 如果资源齐全，再运行一次图片 pipeline smoke test
- `cvkit_package.sh`
  - 对 `cvkit` 执行 `conan create`
  - 如果没有检测到系统 OpenCV 或 ONNX Runtime，会优雅跳过
- `all.sh`
  - 顺序执行上述两条脚本

GitHub Actions 示例：

- `.github/workflows/ci.yml`

当前 CI 的默认假设：

- OpenCV 和 ONNX Runtime 由系统提供
- GStreamer 是可选项，由脚本自动检测
- 如果这些系统依赖缺失，脚本会选择 skip，而不是直接硬失败
