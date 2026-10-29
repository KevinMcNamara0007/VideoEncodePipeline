# EWORLD

Entangled world. This folder is the runnable demo for the EHC aerial encode:
objects from YOLOv8x and YOLO-World, deduplicated across frames, each
object's stencil read by Florence-2 on a few of its frames, and the keyframe
vectors in one index. The object layer is `config/object_layer.json`. The
object layer (worker threads), the tracker and the OCR reader run as
concurrent stages; the reader takes an object the moment its track closes.

EHC stays the library. The recipe pipeline lives in this folder and is
native C++: `src/aerial_encode.cpp` (the encode driver) and
`src/aerial_paint.cpp` (the preview), described in `recipe_aerial_overlay.md`.
`./run.sh` builds them and runs them.

## Run

Put 1–5 minute aerial clips in `input/` (`.mp4`, `.mov`, `.mkv`, `.webm`).

```bash
./run.sh
```

That builds `build/` if the binaries are missing (it needs EHC's
`build/libehc_core.a`, OpenCV, Tesseract and nlohmann-json), exports the YOLO
heads to ONNX the first time, encodes `input/` into `out/`, and writes
`output/preview.mp4`: the survivor keyframes with their boxes drawn, each
labelled `class #object_id  TEXT`. `out/objects.jsonl` is the object index.

Build by hand:

```bash
cmake -S . -B build -DEHC_ROOT=/path/to/EHC -DHALOCORE_ROOT=/path/to/HaloCore
cmake --build build -j
```

Query a token the encoder stored:

```bash
../EHC/build/tools/ehc_video_concept_query out truck --filter-matched
../EHC/build/tools/ehc_video_concept_query out ocr:B127 --filter-matched
```

`./run.sh --smoke` builds a 4 second plate with the stencil `B127` when
`input/` is empty, then encodes that. It checks that the toolchain runs.
COCO `yolov8n` will not label that plate as a tank.

## What is still Python

Only `tools/export_onnx.py`: it converts the Ultralytics checkpoints to ONNX
once per checkpoint (it needs `ultralytics`, `onnx`, and CLIP for
YOLO-World). Nothing in the per-clip path runs Python.

The earlier Python tools (the pass-2 HaloCore front-door scripts, the range
catalog, the Higgsfield clip generators, `florence_gap.py`,
`halocore_florence.py`) were removed; they are in commit `c38c477` if one is
needed again (`git show c38c477:pass2_frontdoor.py`).
