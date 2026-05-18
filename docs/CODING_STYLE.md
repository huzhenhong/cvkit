# Coding Style

This document defines the practical coding-style conventions currently recommended for `cvkit`.

It is intentionally biased toward:

- readability in large C++ codebases
- consistency across public API and internal implementation
- gradual migration instead of large rename-only churn

This is a working engineering guideline, not a rigid purity document. When a local convention would make a file materially harder to read, prefer the simpler, clearer choice and keep the surrounding scope consistent.

## Scope

These conventions apply to:

- public headers under `include/cvkit/`
- internal implementation under `src/`
- examples and test utilities where reasonable

They do not require immediate repository-wide renames. New code should follow them. Existing code should migrate incrementally when a file is already being changed for real work.

## General Principles

- prefer consistency within a module over style churn across unrelated files
- do not make rename-only changes unless they clearly improve a heavily used surface
- avoid broad mechanical renames that hide behavioral diffs
- treat public API naming as more stable than internal helper naming

## Naming Conventions

### Types

Use `PascalCase` for:

- classes
- structs
- enums
- enum classes
- type aliases

Examples:

- `ModelSpec`
- `TensorValue`
- `TaskGraph`
- `CachePolicy`
- `BackendFuture`

### Functions

Use `snake_case` for:

- free functions
- member functions
- static member functions
- namespace-scope helpers

Examples:

- `run_sync()`
- `load_labels()`
- `create_backend_session()`
- `materialize_host_tensor()`
- `build_classification_input()`

Rationale:

- gives the strongest visual separation from `PascalCase` types
- makes function and local-variable naming naturally consistent in implementation-heavy C++ code
- in practice, functions and local variables are still easy to distinguish through:
  - call syntax
  - declaration position
  - surrounding type context

### Local Variables and Parameters

Use `snake_case` for:

- local variables
- function parameters
- loop variables with meaningful names
- intermediate values

Examples:

- `input_shape`
- `cache_path`
- `best_score`
- `device_tensor`

Rationale:

- easy visual distinction from types
- works well in C++ implementation-heavy files

### Member Variables

Use `m_` + `snake_case` for non-static member variables.

Examples:

- `m_ready`
- `m_input_infos`
- `m_execution_stream`
- `m_cache_policy`

Recommended over:

- leading underscore names such as `_ready`
- bare member names that only differ by context

Why `m_`:

- avoids confusion with reserved-identifier patterns involving leading underscores
- is explicit in large implementation files
- aligns well with the rest of the prefix-based rules below

### Global Variables

Avoid global variables unless there is a strong reason.

If a real global variable is necessary, use `g_` + `snake_case`.

Examples:

- `g_default_logger`
- `g_registry`

Rule of thumb:

- prefer local static state, injected context, or object ownership over globals

### Static Variables

Use `s_` + `snake_case` for:

- function-local `static` variables
- file-local static state where a real stateful object is being kept

Examples:

- `s_default_executor`
- `s_empty_labels`
- `s_logger`

### Constants

Use `kPascalCase` for named compile-time constants with stable semantic meaning.

Examples:

- `kDefaultBatch`
- `kMaxPromptPoints`
- `kTensorFileMagic`

For ordinary local `const` values, use normal local-variable naming:

- `const auto max_batch = 1;`
- `const auto cache_root = ...;`

### Macros

Use `ALL_CAPS`.

Examples:

- `CVKIT_WITH_TENSORRT`
- `CVKIT_ENABLE_ONNXRUNTIME`

### Enum Values

Use:

- `PascalCase` for the enum type
- lowercase or `snake_case` for enum values

Examples:

```cpp
enum class MemoryDevice
{
    host,
    cuda,
    npu,
};
```

Do not introduce a second competing style such as `Host` / `Cuda` unless a pre-existing public API already requires it.

## `[[nodiscard]]`

Do not add `[[nodiscard]]` by default.

Use it when ignoring the return value is likely a bug, especially for:

- validation/status functions
- layout/size calculations
- future validity/wait helpers
- result-producing APIs where the return value is the whole point of the call

Examples that usually deserve `[[nodiscard]]`:

- `has_valid_host_layout()`
- `element_count()`
- `packed_byte_size()`
- `run_sync(...)`
- `submit(...)`

Examples that often do not need it:

- simple getters such as `backend()` or `task()`
- interface overrides where the attribute only adds noise
- metadata accessors whose return value is already naturally consumed

## File and API Design Guidance

### Prefer Behavior-Based Names

Good:

- `build_classification_input()`
- `materialize_host_frame()`
- `create_decoder_session()`

Avoid vague helper names like:

- `utils()`
- `commonTool()`
- `helperFunction()`
- `processData()`

### Prefer Narrow Helpers Over Catch-All Utility Dumps

If a helper is:

- reusable across multiple modules
- not tied to example-only behavior
- not tied to a single task family

it may belong in an internal `utils/` area.

If it is:

- example-specific
- output-format-specific
- only there to make a CLI easier to use

it should stay in example support code rather than being promoted too early.

### Avoid Rename-Only Churn

Do not rename files, symbols, or members across wide areas of the repository unless:

- the code is already being touched for real feature or bug work
- the rename materially improves understanding
- the resulting diff still leaves reviewable behavior changes

Good migration pattern:

- new code follows the current convention
- touched files gradually align
- public APIs change only when the benefit is clear

Bad migration pattern:

- repository-wide rename sweep with no behavioral value
- mixing style conversion with large unrelated feature work

## Examples

### Recommended

```cpp
class TensorRuntime
{
  public:
    bool load_model(const ModelSpec& model_spec);
    TaskOutput run_sync(const TaskInput& task_input) const;

  private:
    bool build_engine(const std::string& model_path);

  private:
    std::shared_ptr<IBackendSession> m_backend_session;
    TensorMap                        m_cached_outputs;
    bool                             m_ready{false};
};
```

```cpp
TaskOutput buildClassificationOutput(
    const TensorValue& logits_tensor,
    const std::vector<std::string>& labels)
{
    const auto class_count = logits_tensor.element_count();
    const auto best_index = 0;
    const auto best_score = 0.9F;

    static const std::string s_empty_label{};

    ClassificationValue classification_value{};
    classification_value.class_id = best_index;
    classification_value.score = best_score;
    classification_value.label = class_count > 0 ? labels[best_index] : s_empty_label;
    return {};
}
```

## Migration Strategy

Recommended order:

1. new files follow this document immediately
2. actively modified implementation files migrate opportunistically
3. internal APIs migrate before public APIs
4. public API renames happen only when the benefit outweighs downstream churn

In practice, this means:

- do not stop feature work for broad style-only edits
- do clean up nearby code when a file is already open for meaningful changes
- prefer steady convergence over one-time repository churn

## Current Recommendation Summary

- types: `PascalCase`
- functions: `snake_case`
- locals/parameters: `snake_case`
- members: `m_` + `snake_case`
- globals: `g_` + `snake_case`
- statics: `s_` + `snake_case`
- named constants: `kPascalCase`
- macros: `ALL_CAPS`
- enum types: `PascalCase`
- enum values: lowercase or `snake_case`

This is the default target style for future `cvkit` work unless a module has a stronger established convention that should be preserved for now.
