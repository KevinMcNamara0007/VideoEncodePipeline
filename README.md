# EWORLD

Entangled world. This folder is the runnable demo for the EHC aerial encode:
YOLOv8 boxes, Tesseract stencil reads, and the keyframe vectors in one index.

EHC stays the library. The recipe pipeline lives in this folder:
`recipe_aerial_overlay.md`, `aerial_overlay_encode.py`, and
`aerial_overlay_paint.py`. `./run.sh` builds the bindings and runs those.

## Run

Put 1–5 minute aerial clips in `input/` (`.mp4`, `.mov`, `.mkv`, `.webm`).

```bash
./run.sh
```

That configures `../EHC/build-py` if the `ehc` module is missing, encodes
`input/` into `out/`, and writes `out/preview.mp4`.

Query a token the encoder stored:

```bash
../EHC/build-py/tools/ehc_video_concept_query out truck --filter-matched
../EHC/build-py/tools/ehc_video_concept_query out ocr:B127 --filter-matched
```

`./run.sh --smoke` builds a 4 second plate with the stencil `B127` when
`input/` is empty, then encodes that. It checks that the toolchain runs.
COCO `yolov8n` will not label that plate as a tank.
