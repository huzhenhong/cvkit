# Model Assets

Put local inference models in this directory.

Current convention:

- YOLO11 ONNX model: `assets/models/yolo11n.onnx`

This repository keeps labels under version control, but does not commit heavyweight model binaries.

The example at `examples/pipeline.cpp` looks for:

- `assets/models/yolo11n.onnx`
- `assets/labels/coco80.txt`

You can copy your test model here and run the example without changing code.
