# Recipe — aerial overlay encode

This pipeline lives in EWORLD and is native C++: `src/aerial_encode.cpp`,
`src/aerial_paint.cpp`, and `run.sh`. EHC is the library it links
(`libehc_core.a`).

Encode a directory of aerial clips so the EHC streaming video encoder carries
object class tokens and each object's text in every keyframe vector, and so
the rectangles can be painted back on the survivor frames.

This is the runnable form of the drone-reel demo. The vector stores labels.
The rectangles live in `overlay.jsonl` and the objects in `objects.jsonl`,
because `DetectedTag` keeps `class` and `confidence` only.

Point it at clips you already have (training range, public affairs, or a
generated aerial). One to five minutes is a normal input. The preview is the
survivor keyframes, the moments the encoder kept.

## Objects first, then text

```
survivor frames ──► object layer ──► dedup ──► OCR, once per object ──► index
                   YOLOv8x (COCO)    same object      Florence-2           objects.jsonl
                 + YOLO-World-x      across frames    (HaloCore door)      ▲
                   merged            = object_id                           │
frames.jsonl tags, overlay.jsonl boxes ◄── each box looks its object up ───┘
```

1. **Object layer.** YOLOv8x (COCO) names people and vehicles. Its other
   classes are dropped (`coco.keep`): toilet, suitcase, bench, parking meter
   and the rest are what a closed vocabulary says about a water tower or a
   tank it has no name for. YOLO-World-x answers prompts for what COCO cannot
   name (`world.prompts`: military tank, military helicopter, water tower …),
   each mapped to an index token. Boxes from both heads that overlap
   (`merge_iou`) are one object; the more confident label wins. Every object
   keeps its own box.
2. **Dedup.** Survivors are in time order. An object in the next survivor is
   the same object when its box overlaps (`track.iou`), the two crops look
   alike (hue-saturation histogram correlation ≥ `track.min_appearance`),
   and the two labels share a `track.groups` entry, across up to
   `track.max_gap` survivors it was not seen in. The appearance test is what
   keeps a big box from following the frame across a cut into another shot;
   the groups let truck/tank/boat swaps between heads stay one object. Each
   object gets an `object_id`, unique across the run, and one class: the
   confidence-weighted vote over its frames. Every box shows its object's
   class (`detected` keeps what the head said in that frame).
3. **OCR, per object.** Florence-2 reads each object on up to `ocr.views`
   crops spread over its frames (first, middle, last): the largest view is
   often a side panel, and the lettering is on the tailgate. Classes in
   `ocr.skip` (people) and crops under `ocr.min_side` are not read. The
   door's `Text in the figure:` line gives the stencil words; the caption is
   not used. Scene text is an object class too: Tesseract only *locates*
   text lines (`text`), Florence reads them. A text line inside another
   object's box (`text.host_containment`) is that object's text (AMMO on a
   truck); one with no host (NORTH GATE) is an object of its own.
   **Corroboration:** spellings within edit distance half their length are
   one word (B127 / B27 / 8127), and a word is kept only when two reads
   (Florence views, or Florence and the scout) support it, unless just one
   read saw any text. That is what removes misreads and hallucinations on
   dust.
4. **Index.** `objects.jsonl` maps `object_id` to class, text and the frames
   it is in. Every box in every frame takes its `ocr` from there, so a plate
   read once is on every frame of that object, and the frame's `ocr:` tags
   follow.

All four steps are in `config/object_layer.json`, not in code.

## What each piece does

| Piece | Role in this recipe |
|---|---|
| ffmpeg | Samples RGB frames. Same filter the C++ pipe uses: `fps=N,scale=W:H`. |
| `StreamingVideoEncoder` | pHash dedup, vision features, VSA dedup, then the detector once per video at `finalize()`. |
| YOLOv8x (OpenCV DNN) | COCO people and vehicles. `models/yolov8x.onnx`. Post-processing is Ultralytics predict (letterbox, conf, per-class NMS 0.7); boxes match Ultralytics exactly. |
| YOLOv8x-worldv2 (OpenCV DNN) | The config's prompts, baked into `models/yolov8x-worldv2-aerial.onnx`. |
| HaloCore Florence-2 | `libhalocore_door` (opened at run time from `/opt/mpl/lib`, or `$HALOCORE_DOOR_LIB`) runs Microsoft Florence-2 (`/opt/mpl/models/florence-2-base-onnx`) on the object crops, one door call per video. |
| Tesseract (libtesseract) | The OCR when `--ocr tesseract`: one read per object crop. |
| OpenCV | Reads the survivor PNGs, cuts the crops, draws the preview. |

A YOLO-World `.onnx` answers the prompt list it was exported with, and the
encoder refuses a graph whose prompts differ from the config's. To change the
prompts, edit `world.prompts`, delete `models/yolov8x-worldv2-aerial.onnx`,
and run `./run.sh` (it exports again).

## Why these heads

On 83 frames of `input/drone_raw.mp4` (every frame where `yolov8n` emitted
an odd label or a person inside a truck box, plus every 20th survivor),
checked by eye:

| Head | What it did |
|---|---|
| yolov8n | cow, bus, train, toilet, parking meter on trucks and a water tower; a person on a truck's rear |
| yolov8x | all five trucks of the convoy, the turret gunner, no person on the truck; still train / suitcase / bench on tanks and bridges (no such COCO class) |
| YOLO-World-x, "tank" | missed the tanks even at conf 0.05 |
| YOLO-World-x, "military tank" | the tanks (0.24–0.70), the helicopter (0.84), the water tower ("water tower"/"water tank", 0.2–0.6) |
| YOLO-World-x, "armored personnel carrier" | fired on most cargo trucks (0.2–0.5): left out of the prompts |

The prompt wording matters more than the model size. Confidences of the two
heads are not calibrated against each other; the merge trusts the larger one.

## Class names you will actually get

| You asked about | Token this recipe writes |
|---|---|
| people | `person` (COCO) |
| truck, car, bus, motorcycle, bicycle | the COCO name |
| air vehicles | `airplane` (COCO), `helicopter` (World "military helicopter") |
| boats | `boat` (COCO) |
| tank | `tank` (World "military tank") |
| humvee, artillery | `humvee`, `artillery` (World; rarely fire) |
| water tower, storage tanks | `water-tower`, `storage-tank` |
| weapons | `weapon:rifle`, `weapon:machine-gun` (World) |
| hull numbers, signs | `ocr:B127`, the object's text from the index (A–Z and 0–9, 2–16 chars) |
| scene | one rule tag from the classes in the frame: `scene:air`, `scene:water`, `scene:ground`, `scene:ground-with-person`, `scene:person`, `scene:armor`, `scene:other` |

`scene:armor` is written when the frame's classes include `tank`, `apc`,
`humvee`, or `artillery`.

Text on scenery (a gate sign, a crate) is a `text` object labelled with its
words; text on a detected object is that object's `ocr`.

## 1. Build

The pipeline links EHC's `build/libehc_core.a` and reads HaloCore's door ABI
header (`cpp/src/doc/door_abi.h`). It also needs OpenCV 4 (core, imgproc,
imgcodecs, dnn), Tesseract with Leptonica, and nlohmann-json; Homebrew has
all three.

```bash
cmake --build /path/to/EHC/build --target ehc_core
cmake -S . -B build -DEHC_ROOT=/path/to/EHC -DHALOCORE_ROOT=/path/to/HaloCore
cmake --build build -j
```

`build/` gets `aerial_encode`, `aerial_paint`, and two checks:
`yolo_dump` (a head's boxes, to compare with Ultralytics predict) and
`ocr_dump` (the words Tesseract keeps, to compare with the CLI).

`./run.sh` downloads `yolov8x.pt` and `yolov8x-worldv2.pt` and exports them
once with `tools/export_onnx.py` (needs `ultralytics`, `onnx`, and CLIP for
YOLO-World).

Leave `TESSDATA_PREFIX` unset. libtesseract reads Homebrew's `eng` data.

## 2. Encode

```bash
build/aerial_encode input out --fps 2 --frame 640x640 --ocr florence
```

`--objects` picks another object-layer config; `--model`, `--world-model`
and `--conf` override single entries of it. `--ocr tesseract` reads the
crops with Tesseract instead of Florence; `--ocr none` skips OCR.

`--fps 2` is the demo rate. A locked-off aerial then keeps a handful of
survivor frames instead of every frame. `--frame 640x640` is the YOLO and
OCR input. The visual-only default of 64×64 is too small for either.

Files in `out/`:

| File | Contents |
|---|---|
| `config.json` | Geometry, fps, frame size, the object layer, `detector_token_seed` (2400). |
| `videos.jsonl` | One source per line, including `src_w` and `src_h`. |
| `frames.jsonl` | One survivor per line: `video_id`, `frame_idx`, `ts_ms`, `row_id`, `tags`. |
| `frames.lsh` | The LSH index of the fused frame vectors. |
| `overlay.jsonl` | The boxes `frames.jsonl` does not store, each with its `object_id`. |
| `objects.jsonl` | The object index: one deduplicated object per line. |
| `triples.jsonl` | The clip as triples for the HaloCore ingest front door (section 5). |
| `encode_log.jsonl`, `encode_summary.json` | Counts, including objects and tokens emitted and dropped. |

`ts_ms` is `frame_idx / fps` on the sampled stream. That is the same clock
as `FFmpegFramePipe` in v1. It is the sample clock, not a container PTS.

An `overlay.jsonl` row:

```json
{
  "video_id": 1, "source_path": "/path/to/input/pass.mp4",
  "frame_idx": 40, "ts_ms": 20000.0, "row_id": 3,
  "frame_w": 640, "frame_h": 640, "src_w": 1920, "src_h": 1080,
  "scene": "scene:ground",
  "boxes": [
    {"object_id": 12, "class": "truck", "confidence": 0.86, "xyxy": [80, 120, 410, 380],
     "ocr": ["B127"], "source": "coco"}
  ]
}
```

An `objects.jsonl` row:

```json
{
  "object_id": 12, "video_id": 1, "class": "truck", "source": "coco",
  "confidence": 0.91, "ocr": ["B127"],
  "first_ts_ms": 19500.0, "last_ts_ms": 23000.0,
  "best": {"frame_idx": 44, "ts_ms": 22000.0, "xyxy": [40, 90, 520, 470]},
  "frames": [{"row_id": 2, "frame_idx": 39, "ts_ms": 19500.0}, "..."]
}
```

`xyxy` is in the 640×640 encoder frame. ffmpeg's `scale=W:H` stretches, so
the paint step scales x by `src_w/frame_w` and y by `src_h/frame_h`.

Tags fused into the vector for that frame include `truck`, `ocr:B127`, and
`scene:ground`. The per-frame budget is `max_items_per_frame` (40) minus the
vision features. Lowest confidence is dropped first. OCR tokens are capped
at 8 per frame.

## 3. Query a label

```bash
../EHC/build/tools/ehc_video_concept_query out truck --filter-matched
../EHC/build/tools/ehc_video_concept_query out ocr:B127 --filter-matched
../EHC/build/tools/ehc_video_concept_query out scene:air --filter-matched
```

The concept query builds the same token the encoder superposed
(`feature_to_sparse` at `detector_token_seed` 2400) and ranks videos.
`--filter-matched` keeps a hit only when `frames.jsonl` lists that exact tag.

The query is only as good as the LSH fetch. On `input/drone_raw.mp4`
(326 survivors) the tag is in the vectors: tagged frames score about 0.05
against the token and untagged frames about 0, and for `ocr:B127` and
`person` the two never overlap. But LSH (8 tables, hash 16) hands back 0 to
3 candidates per single-token probe, so most labels return nothing. That is
the index settings, not the encode. `objects.jsonl` answers "which frames
show plate B127" directly.

## 4. Paint the preview

```bash
build/aerial_paint out output/preview.mp4
```

Each survivor is seeked at `ts_ms / 1000` (input seek, after `-i`, so the
frame matches the sample clock closely), boxes are scaled onto the source
frame and labelled `class #object_id  TEXT  confidence`: the same `#id` on
two frames is one object. Survivors play at 2 fps (`--hold 0.5`). A class
keeps its colour from run to run.

## 5. Ingest through the HaloCore front door

`aerial_encode` also writes `out/triples.jsonl`: the clip as triples for the
one ingest door, so the questions in `50 potential questions.md` are asked
of a HaloCore store, not of these files.

```bash
halocore ingest --source out/triples.jsonl --output /path/to/stores/eworld
halocore query --index /path/to/stores/eworld --text "What is the title of frame 226?"
# or: POST /v1/ingest {"source": ".../out/triples.jsonl", "store": "eworld"} on the appliance
```

Subjects are names (`frame 26`, `truck`, `drone raw`, `B127`): a colon id
would make the store record-keyed and cost it its surfaces. The film is
named after its file. Relations are slugs the relation-surface lexicon has
forms for:

| Question words | Triple |
|---|---|
| title, running time, source of description, an aspect of | `<film> title / duration / described_by_source / facet_of` |
| part of, point in time | `frame N part_of <film>`, `frame N point_in_time <ms>` (every sampled frame) |
| also known as | `frame N said_to_be_the_same_as frame M`: a frame the encoder dropped, and the survivor that stands for it |
| story setting, situated at | `frame N narrative_location <scene>`, `<scene> location frame N` |
| shown in, depicted in | `<class> depicted_by frame N`, `frame N depicts <class>` |
| title / on-screen text of a frame, shown in for a plate | `frame N title W`, `frame N subtitle W`, `W depicted_by frame N`, `W subtitle frame N` |
| an attribute of | `frame N facet_of no-box` |

Measured on `input/drone_raw.mp4` (3,330 triples, 383 subjects, every
relation surfaced): of the 75 questions, 57 answer with a fact that is in
the triples and 18 decline. Every decline is a fact that is not there: the
COCO classes the object layer drops (train, dining table, stop sign,
parking meter), the old separate "ocr pass" film, frames with no box
(frame 5), survivors that are no other frame's stand-in, and words not read
(RIDGE, MK82). No question returns a wrong fact. Plate codes (B127, T441,
C330) go through HaloCore's record lane; that needs HaloCore with the
record-lane hop fix (branch `cpp`), or "What is B127 shown in?" answers
`B127`.

## OpenCV as its own detector

A Haar or DNN detector is the pattern in `how_to_use_video_encoding.md`: a
script that prints `yolo_json` and is passed to `ehc_video_encode --detector`.
That path fills `tags` and drops `xyxy`. Use it when you only need the
concept query. Use `aerial_encode` when the reel has to show rectangles.
