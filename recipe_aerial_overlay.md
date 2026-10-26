# Recipe — aerial overlay encode

This pipeline lives in EWORLD: `aerial_overlay_encode.py`,
`aerial_overlay_paint.py`, and `run.sh`. EHC is the library it links.

Encode a directory of aerial clips so the EHC streaming video encoder carries
YOLOv8 class tokens and OCR strings in each keyframe vector, and so the
rectangles can be painted back on the survivor frames.

This is the runnable form of the drone-reel demo. The vector stores labels.
The rectangles live in `overlay.jsonl`, because `parse_yolo_json` and
`DetectedTag` keep `class` and `confidence` only.

Point it at clips you already have (training range, public affairs, or a
generated aerial). One to five minutes is a normal input. The preview is the
survivor keyframes, the moments the encoder kept.

## What each piece does

| Piece | Role in this recipe |
|---|---|
| ffmpeg | Samples RGB frames. Same filter the C++ pipe uses: `fps=N,scale=W:H`. |
| `StreamingVideoEncoder` | pHash dedup, vision features, VSA dedup, then detectors at `finalize()`. |
| YOLOv8n (`ultralytics`) | COCO boxes and class names on the survivor PNGs. Weights `models/yolov8n.pt`. |
| YOLOv8s-world | Weapon prompts only: rifle, machine gun, pistol. Armor names come from the Florence caption. |
| OpenCV | Reads those PNGs, preprocesses stencil crops, draws the preview. |
| HaloCore Florence-2 | `libhalocore_door` runs Microsoft Florence-2 (`/opt/mpl/models/florence-2-base-onnx`) once over the survivor stills. The caption fills classes YOLO did not name. `<OCR_WITH_REGION>` supplies `ocr:<TEXT>`. |
| Tesseract | Second read inside each box, and a full-frame fallback when nothing else found text. |

`ehc_video_encode --detector` can fuse one CLI detector's labels, and it
drops box geometry. Painting the reel needs the Python driver below, which
sees `EncodedFrame.meta` (`video_id`, `frame_idx`, `ts_ms`) and the boxes
from the same `detect()` call.

## Class names you will actually get

`yolov8n.pt` is COCO. Tokens that show up with those weights:

| You asked about | Token this recipe writes |
|---|---|
| air vehicles | `airplane` from COCO, or `helicopter` when the Florence caption says it |
| truck, car | `truck`, `car`, `bus` from COCO |
| tank | `tank` when the caption says tank |
| humvee | `humvee` when the caption says humvee, or when it describes one military vehicle with a machine gun and does not call it a truck or a tank |
| weapons | `weapon:rifle`, `weapon:machine-gun`, `weapon:pistol` |
| people | `person` |
| boats | `boat` |
| bridge | `bridge` when the caption says bridge |
| hull numbers, signs, runway marks | `ocr:B127` from the door line `Text in the figure:`, then Tesseract (A–Z and 0–9, 2–16 chars) |
| scene | one rule tag from the classes in the frame: `scene:air`, `scene:water`, `scene:ground`, `scene:ground-with-person`, `scene:person`, `scene:armor` |

`scene:armor` is written when the frame's classes include `tank`, `apc`,
`humvee`, or `artillery`. COCO never emits those. The HaloCore caption does.
`--florence none` leaves armor to the world checkpoint (`--weapon-model`).

## 1. Build the Python bindings

The driver imports `ehc`. Build it with the same interpreter that has
`ultralytics`, `opencv-python`, and `numpy`. On this machine that interpreter
is `/usr/bin/python3` (3.9), which already imports all three. It still needs
Python development headers or configure fails.

```bash
cd /Users/loki/Quantum_Computing_Lab/EHC
cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release \
  -DEHC_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE=/usr/bin/python3
cmake --build build-py --target ehc_python
```

The native tools (`ehc_video_encode`, `ehc_video_concept_query`) are the
default target. Build them too if you want label queries from the shell:

```bash
cmake --build build-py --target ehc_video_concept_query
```

Leave `TESSDATA_PREFIX` unset. Homebrew Tesseract already has `eng` plus the
`tsv` config. Pointing `TESSDATA_PREFIX` at `/opt/mpl/share/tessdata` hides
that config and OCR returns no boxes.

## 2. Encode

```bash
cd /Users/loki/Quantum_Computing_Lab/EWORLD
/usr/bin/python3 aerial_overlay_encode.py \
  input \
  out \
  --fps 2 \
  --frame 640x640 \
  --model models/yolov8n.pt \
  --conf 0.25 \
  --ocr tesseract
```

`./run.sh` does this, and builds `../EHC/build-py` first when the `ehc`
module is missing. The weights file is `models/yolov8n.pt`.

`--fps 2` is the demo rate. A locked-off aerial then keeps a handful of
survivor frames instead of every frame. `--frame 640x640` is the YOLO and
OCR input. The visual-only default of 64×64 is too small for either.
`--ocr none` skips Tesseract and still encodes YOLO labels.

First run of `yolov8n.pt` downloads the weights.

Files in `out/`:

| File | Contents |
|---|---|
| `config.json` | Geometry, fps, frame size, `detector_cmd`, `detector_token_seed` (2400). |
| `videos.jsonl` | One source per line, including `src_w` and `src_h`. |
| `frames.jsonl` | One survivor per line: `video_id`, `frame_idx`, `ts_ms`, `row_id`, `tags`. |
| `frames.lsh` | The LSH index of the fused frame vectors. |
| `overlay.jsonl` | The boxes `frames.jsonl` does not store. |
| `encode_log.jsonl`, `encode_summary.json` | Counts, including tokens emitted and dropped. |

`ts_ms` is `frame_idx / fps` on the sampled stream. That is the same clock
as `FFmpegFramePipe` in v1. It is the sample clock, not a container PTS.

An `overlay.jsonl` row:

```json
{
  "video_id": 1,
  "source_path": "/path/to/input/pass.mp4",
  "frame_idx": 40,
  "ts_ms": 20000.0,
  "row_id": 3,
  "frame_w": 640,
  "frame_h": 640,
  "src_w": 1920,
  "src_h": 1080,
  "scene": "scene:ground",
  "boxes": [
    {"class": "truck", "confidence": 0.86, "xyxy": [80, 120, 410, 380], "ocr": ["B127"]}
  ]
}
```

`xyxy` is in the 640×640 encoder frame. ffmpeg's `scale=W:H` stretches, so
the paint step scales x by `src_w/frame_w` and y by `src_h/frame_h`.

Tags fused into the vector for that frame include `truck`, `ocr:B127`, and
`scene:ground`. The per-frame budget is `max_items_per_frame` (40) minus the
vision features. Lowest confidence is dropped first. OCR tokens are capped
at 8 per frame so a noisy crop cannot evict the class tags.

## 3. Query a label

```bash
../EHC/build-py/tools/ehc_video_concept_query out truck --filter-matched
../EHC/build-py/tools/ehc_video_concept_query out ocr:B127 --filter-matched
../EHC/build-py/tools/ehc_video_concept_query out scene:air --filter-matched
```

The concept query builds the same token the encoder superposed
(`feature_to_sparse` at `detector_token_seed` 2400) and ranks videos.
`--filter-matched` keeps a hit only when `frames.jsonl` lists that exact tag.

## 4. Paint the preview

```bash
/usr/bin/python3 aerial_overlay_paint.py out out/preview.mp4
```

Each survivor is seeked at `ts_ms / 1000` (input seek, after `-i`, so the
frame matches the sample clock closely), boxes are scaled onto the source
frame, and the OCR string is drawn on the box. Survivors play at 2 fps.
That file is the demo reel of encoded moments.

## HaloCore Florence-2

`./run.sh` calls the door. `HALOCORE_FLORENCE_DIR` defaults to
`/opt/mpl/models/florence-2-base-onnx`, and `DYLD_LIBRARY_PATH` includes
`/opt/bin` so `libhalocore_door.dylib` can see ONNX Runtime. `--florence none`
skips the door and keeps YOLOv8 plus Tesseract.

The door returns a caption, not a rectangle. Class names from that caption
are stored as tags and are not drawn. Stencil text is drawn on the largest
object box when that box is at least 48×48 pixels. A caption-only label does
not invent a box.

## OpenCV as its own detector

A Haar or DNN detector is the pattern in `how_to_use_video_encoding.md`: a
script that prints `yolo_json` and is passed to `ehc_video_encode --detector`.
That path fills `tags` and drops `xyxy`. Use it when you only need the
concept query. Use `aerial_overlay_encode.py` when the reel has to show
rectangles.
