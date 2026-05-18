# Architecture

`cvkit` is the vision/media domain layer above `basekit`.

```text
app
  -> cvkit
    -> basekit
      -> third-party dependencies
```

This document describes the current internal architecture of `cvkit`, with emphasis on:

- `cvkit::infer`
- task graph execution
- backend session design
- task-oriented preprocess/postprocess layering
- the current host-first, device-aware data contract

Related style guidance:

- `docs/CODING_STYLE.md`

It reflects the repository's current implementation state. Planned directions are called out explicitly and should not be read as already implemented behavior.

## Goals

The current architecture is designed to support:

- multiple inference backends
- multiple task families
- graph-aware execution and tracing
- gradual migration toward device-aware data flow
- strict separation between:
  - task semantics
  - backend execution
  - example-only glue

The design explicitly avoids baking third-party runtime types such as ONNX Runtime or TensorRT objects into public headers.

## High-Level Layout

Current major components:

- `cvkit::core`
  - vision-side base types such as `Frame`, `ImageDesc`, `Detection`, and `BBox`
- `cvkit::media`
  - frame input and media backend selection
- `cvkit::image`
  - image operations shared by inference paths
- `cvkit::infer`
  - backend sessions, task graph, task pipelines, async execution, and debug/export helpers

Inside `cvkit::infer`, the layering is:

```text
Model
  -> TaskGraph
    -> Graph Nodes
      -> Task Pipelines / Task Helpers
        -> Backend Sessions
```

More concretely:

- `Model`
  - public entrypoint used by examples, tests, and consumers
- `TaskGraph`
  - internal execution graph with node metadata, explicit/data-driven dependencies, and trace collection
- graph nodes
  - bridge `TaskInput`/`Packet` data into concrete task execution steps
- task pipelines and helpers
  - perform preprocess, task-level orchestration, and postprocess
- backend sessions
  - execute tensor workloads through ONNX Runtime or TensorRT

## Public API Surface

Key public headers:

- `include/cvkit/infer/model.h`
- `include/cvkit/infer/task_io.h`
- `include/cvkit/infer/task_schema.h`
- `include/cvkit/infer/session.h`
- `include/cvkit/infer/tensor_io.h`
- `include/cvkit/infer/debug.h`

The central public object is `cvkit::infer::Model`.

Current public responsibilities of `Model`:

- load a task/backend configuration via `ModelSpec`
- expose backend/session metadata via `session_info()`
- run synchronously via `run_sync(...)`
- run asynchronously via `submit(...)`
- expose graph/debug info for tracing and JSON export

`Model` is intentionally an orchestration layer. It should not grow backend-specific logic or task-specific tensor packing.

## ModelSpec

`ModelSpec` is the main configuration contract for loading a model.

Current important fields include:

- `model_path`
- `aux_model_path`
  - used by split-model flows such as EfficientSAM encoder/decoder
- `backend`
- `task`
- `family`
- `cache_policy`
- `cache_dir`
- `tensorrt_profiles`
- `tensorrt_prefer_device_outputs`

Usage pattern:

- detection / YOLO11
  - `model_path` points to the detector model
- promptable segmentation / EfficientSAM combined
  - `model_path` points to the encoder
  - `aux_model_path` points to the decoder
- promptable segmentation encoder-only
  - `family = efficient_sam_encoder`
- promptable segmentation decoder-only
  - `family = efficient_sam_decoder`

## Task Families

Current first-stage implemented tasks:

- `detection`
- `classification`
- `promptable_segmentation`

Planned but not yet implemented task families:

- `segmentation`
- `pose`
- `facemesh`

The current architecture is intentionally set up so that adding a new task family should primarily involve:

- task schema definition
- preprocess/postprocess logic
- graph node wiring

and should not require redesigning backend sessions.

## Backends

Current implemented backends:

- `onnxruntime`
- `tensorrt`

Reserved but not implemented:

- `openvino`
- `ffmpeg` for media, not inference

### Backend Responsibilities

Backend sessions are responsible for:

- loading backend-native model/runtime state
- exposing input/output tensor metadata
- running backend tensor execution
- optionally supporting backend-aware async execution

Backend sessions are not responsible for:

- task-level image preprocessing semantics
- detection postprocess
- promptable segmentation prompt-building semantics
- example formatting or file export

### Backend Session Contract

Internal backend contract lives in:

- `src/infer/backends/backend_session.h`

Current key pieces:

- `IBackendSession`
- `RawTensor`
- `RawTensorMap`
- `BackendFuture`

`RawTensor` currently aliases the public `TensorValue`, and `RawTensorMap` aliases the public `TensorMap`.

This is deliberate: the codebase is converging toward a single tensor contract instead of maintaining parallel public and internal tensor types.

### Current Backend Execution Constraints

The current execution contract is intentionally narrow:

- backend input tensors must be:
  - `float32`
  - layout-valid
- current generic path expects host-accessible tensors
- current TensorRT path additionally supports CUDA-resident external-view float32 tensors for direct input binding
- current TensorRT path can also prefer CUDA-resident raw outputs, while task-level postprocess remains free to materialize host copies if needed
- backend output export currently supports:
  - `float32` only

This is stricter than the metadata contract:

- session metadata may describe richer dtypes
- runtime execution/export still only supports the subset actually implemented

This split is intentional and keeps metadata evolution ahead of execution support without lying about runtime capability.

## Task Graph

The internal graph implementation lives in:

- `src/infer/graph/graph.h`
- `src/infer/graph/graph.cpp`
- `src/infer/graph/packet.h`

The graph is not rebuilt per internal step of a task. Instead:

- a graph structure is created per model/task configuration
- a `Packet` flows through that graph
- nodes perform task steps

### Graph Responsibilities

The current graph supports:

- named nodes
- explicit dependencies
- inferred data dependencies from `produces()` / `consumes()`
- graph boundary derivation
- synchronous packet execution
- async packet submission
- per-node trace and timing

### Packet

`Packet` is the internal carrier for:

- `TaskInput`
- `TaskOutput`
- task-local scratch values
- trace metadata

`Packet` should be treated as graph-internal transport, not as a public consumer API.

### Graph Boundaries

Current graph metadata exposes:

- nodes
- dependencies
- consumed keys
- produced keys
- boundary inputs
- boundary outputs

This is used by:

- `--print-graph`
- `--dump-graph-json`
- debug helpers in `cvkit::infer::debug`
- graph JSON export currently uses schema version `5`

### Graph Async Model

The current graph has a minimal node-level async contract:

- `INode::supports_async()`
- `INode::submit(Packet)`

Today this is used by:

- detection infer node
- promptable segmentation infer node

This is enough to support graph-aware async without committing yet to a full streaming runtime or graph-wide scheduler.

## Task Pipelines

Task-specific code lives under:

- `src/infer/tasks/detection/...`
- `src/infer/tasks/promptable_segmentation/...`

The intended split is:

- task family code owns task semantics
- backend sessions own tensor execution

### Detection

Current detection implementation is YOLO11-oriented.

Key pieces:

- YOLO preprocess on CPU
- backend tensor execution
- YOLO postprocess
- graph nodes:
  - infer
  - postprocess

Detection async behavior currently benefits from:

- graph-aware async
- TensorRT backend-aware async when available

Detection preprocess is now device-aware at the dispatch layer:

- host-backed `ImageValue`
  - uses the existing CPU preprocess path
- CUDA-backed `ImageValue`
  - is recognized as a distinct preprocess source
  - currently runs either a device-side kernel or a device-to-host-copy fallback, depending on build capability
  - on the TensorRT path, the device-side kernel can now feed a CUDA-resident float32 tensor directly into backend execution

### Promptable Segmentation

Current promptable segmentation implementation is EfficientSAM-oriented.

Supported model families:

- `efficient_sam`
- `efficient_sam_encoder`
- `efficient_sam_decoder`

Supported modes:

- combined
- encoder-only
- decoder-only

Current outputs include:

- `mask`
- `scores`
- `low_res_masks`
- `logits`
- `image_embeddings`

### Classification

Current classification support is intentionally minimal and host-first.

Current behavior:

- input:
  - `image`
- preprocess:
  - host-side image materialization when needed
  - resize
  - RGB conversion
  - `[0, 1]` normalization
  - NCHW `float32` tensor packing
- outputs:
  - `classification`
  - `scores`

Current intent:

- validate that the existing:
  - task schema
  - pipeline
  - backend session
  - task I/O value model
  can support a third task family without structural redesign

Current limitation:

- there is not yet a bundled classification asset/example path comparable to the YOLO11 and EfficientSAM flows
- repository coverage currently comes from focused unit tests and stub backend sessions

Promptable segmentation is already graph-integrated and supports the same graph-aware async model as detection, but its implementation remains first-stage and task-specific.

## Data Contract

The current data contract centers around:

- `TaskInput`
- `TaskOutput`
- `Value`

Key value types currently include:

- `ImageValue`
- `MaskValue`
- `BoxListValue`
- `KeypointsValue`
- `TensorValue`
- detection lists
- point lists
- strings
- float lists

### ImageValue

`ImageValue` is the current device-aware image carrier.

It currently holds:

- `frame`
- `memory_device`
- `device`
- `storage`
- `row_stride_bytes`
- `external_data`
- `storage_bytes`

Current helper semantics:

- `is_host_accessible()`
- `bytes_per_pixel()`
- `packed_row_stride_bytes()`
- `effective_row_stride_bytes()`
- `is_packed()`
- `has_valid_host_layout()`
- `required_byte_size()`
- `has_valid_device_view()`

Current implementation status:

- examples and task pipelines use host-owned image values
- `StorageKind::external_view` now has a real first-stage use:
  - detection can consume external CUDA image views through `external_data`
- current detection CUDA path has two execution modes:
  - device-side YOLO preprocess kernel when CUDA language is enabled
  - fallback device-to-host copy followed by existing CPU preprocess when only `cudart` is available
- true zero-copy end-to-end image handling is still not implemented

### TensorValue

`TensorValue` is the current public tensor carrier.

It currently holds:

- `name`
- `shape`
- `data`
- `data_type`
- `memory_device`
- `storage`

Current helper semantics:

- `is_host_accessible()`
- `element_count()`
- `packed_byte_size()`
- `byte_size()`
- `is_packed()`
- `has_valid_host_layout()`
- `owns_storage()`

Current implementation status:

- tensor storage is still host-side `std::vector<float>`
- non-float32 metadata is supported
- non-float32 execution is not yet supported
- `external_view` remains a forward-looking contract for tensors
- promptable segmentation already uses this contract to accept CUDA-resident decoder embeddings and materialize them back to host when needed
- promptable segmentation encoder and combined paths now also accept CUDA-resident `ImageValue` inputs and materialize them back to host before the current ONNX Runtime encoder path
- when an ONNX Runtime session is explicitly loaded for CUDA execution, promptable decoder embeddings can now also enter ORT through a CUDA-resident tensor input path; image inputs for the encoder path remain host-materialized today

### StorageKind

`StorageKind` currently distinguishes:

- `owned`
- `external_view`

Today:

- current examples and runtime-produced tensors/images are `owned`
- `external_view` exists so the public contract can evolve toward:
  - zero-copy host views
  - external GPU-backed buffers
  - borrowed memory contracts

without breaking type structure later

## Session Metadata

Public session metadata lives in:

- `include/cvkit/infer/session.h`

Key pieces:

- `TensorInfo`
- `SessionInfo`

Current metadata includes:

- tensor name
- shape
- data type
- memory device

The metadata contract is already richer than the execution contract and is exposed through:

- `Model::session_info()`
- debug print output
- graph JSON export

## Tensor File I/O

Tensor file helpers live in:

- `include/cvkit/infer/tensor_io.h`
- `src/infer/utils/tensor_io.cpp`

Current file format behavior:

- writer emits:
  - `CVKTEMB3`
- reader accepts:
  - `CVKTEMB3`
  - `CVKTEMB2`
  - `CVKTEMB1`

Current stored metadata includes:

- tensor name
- shape
- data type
- memory device
- storage kind

This is currently used by the EfficientSAM example flow for encoder/decoder tensor exchange.

## Debug and Observability

Public debug helpers live in:

- `include/cvkit/infer/debug.h`
- `src/infer/utils/debug.cpp`

Current observability features:

- graph node listing
- graph boundary listing
- per-node trace printing
- session tensor metadata printing
- graph JSON export

Current graph JSON includes:

- task
- backend
- family
- model path
- labels path
- cache policy
- cache dir
- async flag
- graph node metadata
- graph boundary
- trace
- session tensor metadata

Environment/runtime controls:

- `CVKIT_EXECUTOR_THREADS`
- `CVKIT_TRACE_GRAPH=1`

## Examples vs Library Internals

The repository deliberately separates:

- reusable library internals
- example-only helpers

Example-only helpers live under:

- `examples/*_support.cpp`
- `examples/example_*.cpp`

These are allowed to contain:

- CLI-specific orchestration
- output file naming
- example JSON decorations
- visualization helpers

They should not become the dumping ground for reusable inference/runtime logic.

When helper logic becomes genuinely reusable across tasks or examples, it should move into:

- `src/infer/utils/...`
- or a public header if it becomes consumer-facing API

## Current Constraints

This is the most important "do not assume more than this" section.

Current repository state is strongest in:

- YOLO11 detection
- first-stage classification pipeline structure
- EfficientSAM promptable segmentation
- ONNX Runtime + TensorRT backend coverage
- graph execution and observability

Current important constraints:

- image/tensor device awareness is still mostly metadata-first
- host execution remains the dominant real path
- backend tensor execution still only accepts/export `float32` in practice
- no production-grade zero-copy or external-view memory contract exists yet
- GPU preprocess is currently detection-only and still partly fallback-based
- no generalized streaming media graph runtime exists yet

## Planned Evolution

Planned near-term directions:

- continue device-aware data-path work
- broaden device-side preprocess beyond the current detection-first path
- continue tightening backend tensor-engine contracts
- extend task coverage:
  - segmentation
  - pose
  - facemesh

Planned later directions:

- real external-view / zero-copy contracts
- broader non-host tensor execution support
- broader non-float32 runtime support
- stronger streaming/runtime orchestration where it becomes justified by real media throughput needs

Important note:

- these are architectural directions, not guarantees of current production readiness
