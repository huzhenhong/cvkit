# cvkit

`cvkit` 是位于 `basekit` 之上的视觉/媒体领域库。

架构细节：

- `docs/ARCHITECTURE.md`
- `docs/CODING_STYLE.md`

分层关系：

```text
app
  -> cvkit
    -> basekit
      -> third-party dependencies
```

当前阶段重点：

- 基于 ONNX Runtime 和 TensorRT 的 YOLO11 detection
- 具备 encoder / decoder 分离能力的 EfficientSAM 风格 promptable segmentation
- 可切换后端的图像/视频输入
- graph-aware 的异步执行、tracing 和 JSON 导出
- 保持领域 API 与第三方实现之间的边界清晰

## 组件

- `cvkit::core`
  - 视觉侧共享类型与稳定基础定义
- `cvkit::media`
  - 媒体后端选择与帧输入
- `cvkit::image`
  - 推理流程所需的图像处理
- `cvkit::infer`
  - 多后端推理、任务图执行与任务导向 pipeline

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

当前第一阶段推理范围：

- 已实现任务：
  - detection
  - classification
  - promptable segmentation
- 已实现 backend：
  - onnxruntime
  - tensorrt
- 已规划但尚未实现的任务族：
  - segmentation
  - pose
  - facemesh

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

当前 promptable segmentation 相关资源：

- encoder 模型：
  - `assets/models/efficient_sam_vitt_encoder.sim.onnx`
- decoder 模型：
  - `assets/models/efficient_sam_vitt_decoder.sim.onnx`

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
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- 把当前配置的 `compile_commands.json` 同步到 `build/compile_commands.json`

当前脚本行为说明：

- 默认模式是 `release`
- 每次成功构建后，活动构建树里的 `compile_commands.json` 会刷新到：
  - `build/compile_commands.json`
- TensorRT 相关 smoke 仍然是显式启用：
  - `CVKIT_RUN_TENSORRT_SMOKE=1`

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
- `CVKIT_EXECUTOR_THREADS`
- `CVKIT_TRACE_GRAPH`

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

TensorRT cache 规则：

- 默认 cache 根目录：
  - `assets/models/.cvkit_cache/tensorrt/`
- 默认策略：
  - `default`
- cache 文件命名：
  - 模型指纹 + 运行时指纹
- 旧的 `*.trt.plan` 仍然兼容，命中后会尽量迁移到新的 cache 布局
- `ModelSpec::tensorrt_profiles`
  - 可选的按输入张量覆盖 TensorRT optimization profile shape 的能力
  - 可以按输入名显式提供 `min/opt/max` shape
  - 如果不提供，`cvkit` 会对当前 detection 和 promptable segmentation 主线使用内置默认策略
- `ModelSpec::tensorrt_prefer_device_outputs`
  - 可选的 TensorRT 专属 hint，用于在可行时让 backend output metadata 和原始输出保持 CUDA-resident
  - 当前 detection 路径在开启该选项时仍可正常运行，因为 detection postprocess 会在需要时自行 materialize host copy

当前公共数据 contract 说明：

- `TaskInput` / `TaskOutput`
  - 已支持比裸 `Frame` 和检测结果列表更丰富的 value 类型
- `ImageValue`
  - 当前包含：
    - `frame`
    - `memory_device`
    - `device`
    - `storage`
    - `row_stride_bytes`
    - `external_data`
    - `storage_bytes`
  - 当前 examples 和 host 侧 pipeline 使用的都是 `StorageKind::owned`
  - host 侧处理目前要求图像具备有效的 host layout
  - detection 现在也接受 `memory_device=cuda` 的图像输入，但要求提供有效的外部 device view
- `TensorValue`
  - 当前包含：
    - `name`
    - `shape`
    - `data`
    - `data_type`
    - `memory_device`
    - `storage`
    - `external_data`
    - `storage_bytes`
  - 当前通用执行层 contract 接受 host `float32` 输入 tensor
  - 当前 TensorRT 路径还额外支持 CUDA-resident external-view `float32` 输入 tensor
- classification 第一阶段状态：
  - 当前 pipeline 是 host-first、task-oriented 的最小实现
  - 输入图像会在 CPU 侧完成 resize 和 normalize
  - 当前输出：
    - `classification`
    - `scores`
  - 仓库里目前还没有内置 classification 模型资源，因此当前覆盖主要来自聚焦单测和 stub backend
- promptable segmentation decoder 现在也可以消费 CUDA-resident 的 `image_embeddings`，并在进入当前 ONNX Runtime 执行路径前 materialize 回 host
- promptable segmentation 的 encoder 和 combined 路径现在也可以接受 CUDA-resident 的 `ImageValue` 输入；当前 ONNX Runtime 执行仍会先 materialize 回 host 再喂 encoder
- 当 ONNX Runtime session 通过 `ModelSpec.device.kind = cuda` 加载时，promptable decoder 的 CUDA-resident embeddings 现在也可以走真正的 ORT CUDA tensor 输入路径，而不再总是先回拷到 host
  - 当前 backend 导出 contract 仍然只输出 `float32` tensor
- tensor 文件格式：
  - 当前写出格式是 `CVKTEMB3`
  - 读取仍向后兼容 `CVKTEMB2` 和 `CVKTEMB1`

当前 detection 的 CUDA preprocess 状态：

- 如果 configure 阶段可用 CUDA language，`cvkit::infer` 会编译 device-side YOLO preprocess kernel
- 如果当前只有 `cudart` 可用，detection 仍可接受 `ImageValue(memory_device=cuda)`，但当前会回退为：
  - device-to-host copy
  - 现有 CPU preprocess
- 在启用 CUDA language 的 TensorRT 路径上，detection 现在还能直接产出 CUDA-resident 的 float32 输入 tensor，并直接绑定给 TensorRT，不再额外做一次 host-to-device upload
- 当前 CMake 自动探测会优先尝试：
  - `/usr/local/cuda-13.0/bin/nvcc`
  - 然后 `/usr/local/bin/nvcc`

执行与图 tracing：

- `--async`
  - 通过 `Model::submit()` 走异步推理路径，而不是直接同步调用
- `--print-graph`
  - 在 example 进程里直接打印任务图节点、推导出的图边界，以及最近一次逐节点 trace
  - 适合检查 `consumes` / `produces` 数据契约和实际节点顺序，不需要额外挂调试器
- `--dump-graph-json`
  - 将同一份图 metadata 和最近一次 trace 写到 JSON 文件
  - 适合脚本消费、可视化，以及离线查看 DAG 结构
  - 当前 schema version：`5`
- `CVKIT_EXECUTOR_THREADS`
  - 控制内部共享 executor 线程池大小
- `CVKIT_TRACE_GRAPH=1`
  - 通过 `basekit::log` 打开每个图节点的 timing 日志
  - 每条节点日志会包含：
    - 节点名
    - 顺序索引
    - 微秒级耗时
    - 输入数量
    - output 增量
    - scratch 增量

`--print-graph` 输出示例：

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

JSON 导出示例：

```bash
CUDA_VISIBLE_DEVICES=7 build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend tensorrt \
  --image assets/images/test_001.jpg \
  --output-dir assets/output \
  --async \
  --dump-graph-json assets/output/graph.json
```

图片推理示例：

```bash
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend onnxruntime \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --output-dir /workspace/cvkit/assets/output
```

GStreamer 视频读写加 H.264 MP4 输出示例：

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

带显式 cache 控制的 TensorRT 示例：

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

带异步推理和 graph timing 的 TensorRT 示例：

```bash
CVKIT_TRACE_GRAPH=1 CUDA_VISIBLE_DEVICES=7 \
/workspace/cvkit/build/conan/Release/examples/bin/cvkit_example_pipeline \
  --infer-backend tensorrt \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --output-dir /workspace/cvkit/assets/output \
  --async
```

当前工作区里已经生成过的验证输出：

- `assets/output/test_001_det.jpg`
- `assets/output/test_det.avi`
- `assets/output/test_det.mp4`

## Promptable Segmentation 示例

示例可执行文件：

- `build/conan/Release/examples/bin/cvkit_example_promptable_segmentation`

当前这版第一阶段实现：

- backend：
  - `onnxruntime`
- family：
  - `efficient_sam`
  - `efficient_sam_encoder`
  - `efficient_sam_decoder`
- 模型布局：
  - `--encoder` 指向 `efficient_sam_vitt_encoder.sim.onnx`
  - `--decoder` 指向 `efficient_sam_vitt_decoder.sim.onnx`
- 运行模式：
  - `--mode combined`
    - encoder + decoder 端到端联跑
  - `--mode encoder`
    - 只跑 encoder，并导出 `image_embeddings`
  - `--mode decoder`
    - 读取已导出的 `image_embeddings`，只跑 decoder
- 当前支持的 prompt：
  - 通过 `--point-x/--point-y` 提供单点或多点点提示
  - 通过 `--use-box --box-x --box-y --box-w --box-h` 提供可选 box prompt
- embedding 交换：
  - 使用 `--embeddings <path>`
  - 当前 example 使用紧凑的 cvkit 二进制 tensor 文件（`*.bin`）

运行示例：

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

仅 encoder 导出：

```bash
/tmp/cvkit-ort-out/conan/Release/examples/bin/cvkit_example_promptable_segmentation \
  --mode encoder \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --point-x 640 \
  --point-y 360 \
  --embeddings /workspace/cvkit/assets/output/test_001_sam_embeddings.bin
```

仅 decoder 运行：

```bash
/tmp/cvkit-ort-out/conan/Release/examples/bin/cvkit_example_promptable_segmentation \
  --mode decoder \
  --image /workspace/cvkit/assets/images/test_001.jpg \
  --point-x 640 \
  --point-y 360 \
  --embeddings /workspace/cvkit/assets/output/test_001_sam_embeddings.bin \
  --output-dir /workspace/cvkit/assets/output
```

当前工作区里已经生成过的验证输出：

- `assets/output/test_001_sam_mask.png`
- `assets/output/test_001_sam_overlay.png`
- `assets/output/efficient_sam_graph.json`
- `assets/output/test_001_sam_embeddings.bin`

## 当前状态

已在本地验证：

- Conan install 正常
- `cvkit` configure / build 正常
- 本地 tests 通过
- YOLO11 图片推理链路正常
- GStreamer 视频读取正常
- GStreamer `jpegavi` 和 `x264mp4` 写出链路正常
- `CVKIT_ENABLE_GSTREAMER_CUDA=ON` 的 configure / build 路径已经验证
- TensorRT backend 的 load/run/cache 链路已经在第 7 张卡上验证通过
- TensorRT serialized engine cache 已支持指纹命名和旧 cache 迁移
- 内部异步 executor 路径已经验证
- detection 和 promptable segmentation 已接入 graph-aware async 执行路径
- task graph metadata 与每节点 timing trace 已可用
- `cvkit::infer::Model::session_info()` 现在已能通过公共 API 暴露 backend tensor 输入/输出 metadata
- `ImageValue` 和 `TensorValue` 现在都带有最小 device-aware contract metadata
- examples 现在已经使用 `ImageValue` 作为主要图像输入对象
- tensor session/debug metadata 现在会带出：
  - `data_type`
  - `memory_device`
- image/tensor contract 现在已经区分：
  - host layout validity
  - storage kind（`owned` vs `external_view`）

当前一个本地使用注意点：

- 有些本地构建树可能是 `CVKIT_ENABLE_ONNXRUNTIME=OFF`
- 这类构建树下运行 example 时，应显式传 `--infer-backend tensorrt`，不要依赖默认 backend

尚未定稿：

- FFmpeg backend
- OpenVINO backend
- 从 example 支持层抽离成正式生产级 writer 抽象
- 更通用的非 host / 非 float32 tensor 执行路径
- 正式的 GPU preprocess 与更完整的 device-resident 数据流
- detection / classification / promptable segmentation 之外的更多任务族
- 超出当前 metadata-only `storage` contract 的更强 external-view / zero-copy 所有权模型

## 下一步计划

基于当前代码状态，近期计划主要包括：

- 继续推进 device-aware 数据路径
  - 在保持 public contract 稳定的前提下，为 CUDA-backed image/tensor 路径做准备
- 从“只有 metadata 感知”逐步推进到真实的 GPU preprocess 支持
- 继续收紧 backend tensor-engine contract
  - 尤其是 non-host 和 non-float32 执行支持
- 扩展任务覆盖范围
  - segmentation
  - pose
  - facemesh

范围说明：

- 这些是计划方向，不代表当前已经具备 production-ready 状态
- 当前仓库最成熟的部分仍然是：
  - YOLO11 detection
  - EfficientSAM promptable segmentation
  - task-graph 执行与可观测性

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
