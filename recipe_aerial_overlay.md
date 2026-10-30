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
survivor frames ──► object layer ──► dedup ──► OCR, per closed object ──► index
                   YOLOv8x (COCO)    same object      Florence-2             objects.jsonl
                 + YOLO-World-x      across frames    (HaloCore door)        ▲
                   merged            = object_id      on the reader thread   │
                   2 worker threads  calling thread                          │
frames.jsonl tags, overlay.jsonl boxes ◄── each box looks its object up ─────┘
```

The three stages run at once: the workers take frames in order, the
tracker consumes them as they land, and an object whose track has closed
(not seen for `track.max_gap + 1` frames) goes to the reader immediately,
while the layer is still on later frames. The reader batches `ocr.batch`
views per door call.

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
| HaloCore Florence-2 | `libhalocore_door` (opened at run time from `/opt/mpl/lib`, or `$HALOCORE_DOOR_LIB`) runs Microsoft Florence-2 (`/opt/mpl/models/florence-2-base-onnx`) on the object crops, one door call per `ocr.batch` views, on the reader thread. The encode sets `HALOCORE_FIGURES_TASKS=ocr` (the door's object labels and caption are not used) and `HALOCORE_FIGURES_THREADS` to half the cores, unless set already. |
| Tesseract (libtesseract) | Locates scene text on every frame (the scout, inside each worker). The OCR itself when `--ocr tesseract`: one read per object crop. |
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
`--threads N` is the number of object-layer workers (default 2 on 8+
cores): each owns its two YOLO heads and a Tesseract, and a YOLO forward
already spreads over OpenCV's thread pool, so more than 2 is slower
(section "Where the time goes").

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

## Where the time goes

Measured on `input/drone_raw.mp4` (338 frames at 2 fps, 326 survivors, 198
objects, 192 OCR views), 16-core Apple silicon, OpenCV DNN on the CPU. The
numbers are the `detect seconds` line the encode prints (also in
`encode_log.jsonl` as `detect_seconds`).

**Before the stages ran concurrently** (one thread, OCR after the layer,
the door describing every crop three ways): 355 s wall.

| Stage | seconds |
|---|---|
| YOLOv8x COCO, 326 forwards | 71.6 |
| YOLO-World-x, 326 forwards | 80.8 |
| Tesseract scout, 326 frames | 12.9 |
| Florence door, 192 views (object labels + OCR + caption, 4 threads) | 174.8 |

So the door was half the time, not 99 %, and most of its half was the
caption and object labels the encode never reads.

**The same 192 crops through each reader** (`build/ocr_bench`; the crops
come from `EWORLD_KEEP_VIEWS=<dir>`):

| Reader | s / view | views with text | scene words found (of 15) | junk words |
|---|---|---|---|---|
| Tesseract psm 7 | 0.01 | 17 | 5 | 2 |
| Tesseract psm 6 | 0.02 | 30 | 6 | 10 |
| Tesseract psm 11 | 0.01 | 29 | 8 | 5 |
| Tesseract psm 7, crop ×2 | 0.03 | 19 | 6 | 2 |
| door, all three generations, 4 threads | 0.91 | 117 | 15 | 31 |
| door, `HALOCORE_FIGURES_TASKS=ocr`, 4 threads | 0.58 | 117 | 15 | 31 |
| door, OCR only, 8 threads | 0.45 | 117 | 15 | 31 |

The OCR-only door returns the same words as the full pass, in half the
time. Tesseract is 50–100× cheaper and reads the printed signs (NORTH GATE,
FUEL, 762, the runway 27) but none of the stencils on vehicles: B127 comes
out as B27 / B12 / 127, T441 as 144, AMMO as ANNO, and C330, A22, P19,
AF218, MK82, RANGE do not come out at all. Tesseract is a document reader:
it binarises and expects lines of print on a flat page. A stencil on a
curved, lit, foreshortened tailgate in a drone shot is scene text, which
is what Florence-2 was trained on. End to end, `--ocr tesseract` finishes
in 143 s and finds 3 of the 15 words exactly; Florence finds 13 (MK82 is
read on one view only, and the runway 27 as 127).

**Object-layer workers** (`--threads`, `--ocr none`): 1 worker 153 s,
2 workers 126 s, 4 workers 136 s. A forward already spreads over OpenCV's
pool; the second frame in flight fills the gaps, a third and fourth fight
for cores.

**After** (2 workers, the reader on its own thread with
`HALOCORE_FIGURES_TASKS=ocr`, 4 threads, batches of 32): **188 s wall**,
identical `objects.jsonl`. The layer took 154 s (slower than its solo 126 s:
the reader's threads share the cores), the reads 120 s of which all but
the last batch (20 s) ran under the layer. The encode is now YOLO-bound;
the next lever is a faster head, not the OCR.

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

Each source is decoded once, through the same `FFmpegFramePipe` and `fps`
filter the encoder sampled it with (`config.json` `target_fps`), at source
size: survivor `frame_idx` N is frame N of that stream, the exact frame the
boxes were found on. Boxes are scaled onto it and labelled
`class #object_id  TEXT  confidence`: the same `#id` on two frames is one
object. A class keeps its colour from run to run. A box the encoder
withheld from the lake (`fact: false`, section 5) is drawn grey, thin and
labelled `withheld`, so the preview shows what the appliance was not told.
On `drone_raw.mp4` (326
survivors) this takes 8 s; the earlier one-seek-per-survivor version
decoded from the start of the file for every survivor and took 3 min 52 s.

The preview runs on the **source clock**: each survivor stays up until the
next survivor's timestamp (a frame the encoder dropped as a near-duplicate
leaves its stand-in on screen for its slot), the last one for `--hold`
seconds. Preview second `t` is source millisecond `1000 t`, so a fact's
`point_in_time` seeks the preview with no lookup; `drone_raw.mp4` gives a
169 s preview of 326 frames. The video's timestamps are in milliseconds
(`-video_track_timescale 1000`): on ffmpeg's default 1/25 s track clock a
half-second cue landed at 0.52 s, and a seek to "cue plus a little" sat on
the frame boundary.

Beside the preview the step writes `preview.index.json`, the lookup the
appliance's page reads to play the frames a fact names: `frames` (every
survivor in preview order: `frame_idx`, `ts_ms`, `t` and `end` on the
preview clock, `scene`, and its boxes with `object_id`, `class`, `ocr`),
`objects` (objects.jsonl, each with the `frames` it was seen on) and
`videos` (source, sizes, the preview span of each). Section 6 puts both
beside the store.

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

**Only a corroborated object becomes a fact.** The heads run low (COCO
0.25, World 0.20) so that tracks stay continuous across frames, and the
lake scores every triple of a subject alike, so without a gate the weakest
detection in the clip is the first answer: "What is the tank shown in?"
led with frame 102, a 0.26 "tank" box on the A14 truck, followed by a 0.24
sliver at the top edge of frame 146, with the real tank (frames 151 to 175,
0.50 to 0.76) behind them. The rule is the one `corroborate()` already
applies to OCR words: an object is a fact when it was seen on
`objects.min_frames` survivors (2), or once at or above
`objects.min_confidence` (0.4), in `config/object_layer.json`. Measured on
`drone_raw.mp4`: every false tank and humvee was one look under 0.30; the
real objects a single look caught were 0.47 and up (a soldier in a cab, a
distant truck), so 0.5 would have cost them. A withheld object stays in
`objects.jsonl` and `overlay.jsonl` with `fact: false` (the preview paints
it grey), and a frame whose only boxes were withheld is `facet_of no-box`.
With the floor: 14 tank objects become 6 (five real, one 0.47 shot of a
truck's underside), the three "humvee" objects go, 30 objects of 198 are
withheld in all and the clip's triples go from 3,330 to 3,266.

**Corrections.** What the heads cannot get right, an analyst can say once:
`config/corrections.json` (or `--corrections PATH`, `none` to skip) lists
objects by clip stem and `object_id` with a new `class`, or `drop: true`
for a duplicate detection of something already tracked. The encoder
applies them after dedup and before anything is written, so
`objects.jsonl` (`source: manual`, the heads' label kept in `detected`),
the overlay, the painted preview, the index sidecar and the triples all
carry the correction, and a corrected object counts as corroborated. An
`expect: {class, first_frame}` guards each entry: object ids are
deterministic for one object layer on one clip and renumber when the layer
changes, and a guard that no longer holds skips the entry with a message
instead of relabelling the wrong thing. The first use: the M1 Abrams T441
close-up is "truck" to COCO at 0.90 head-on (object 149, frames 229 to
235) and "boat" at 0.86 from the side (object 151, frames 239 to 260) while
the World "military tank" prompt reaches 0.32, so the class vote lost both
times; the corrections make both `tank` and drop the two one-look
duplicate boxes on the second (152, 156). "What is T441 shown in?" and
"What is the tank shown in?" then both reach it. This is a per-clip
override, not a fix for the detector; the detector fix stays on the open
list.

**Near.** The lake is an (subject, relation) lookup; "find all people near
a tank" is a conjunction it cannot form at query time, and with no `near`
in the store it declined ("names something the fabric has no relation
for"). The encoder has the boxes, so it states the geometry: two
corroborated facts in one frame are near each other when their boxes, each
grown by `objects.near_reach` (0.5) of its own width and height on every
side, overlap. Geometry, not the same shot: on drone_raw the FUEL stencil
shares frames 318 to 320 with people who stand by another truck and is
near none of them (same-frame would say 3, nearness says 0). For each near
pair the frame depicts `person near tank` as a subject of its own
(`depicted_by` the frame, so the five-frame player works unchanged), and
the clip states `person near tank` / `tank near person` once. A hosted
stencil is its vehicle's word, so the pair is the person and the word on
the truck they stand by (`person near AMMO`), never a truck and its own
lettering. Measured: 110 near pairs, triples 3,266 to 4,032; `person near
tank` in frames 244 to 260 (12, the crew at T441), `person near truck` in
71 frames, `person near FUEL` in none.

The lake matches a question's span against a subject's surfaces as typed,
so `people near a tank` reached nothing while `person near tank` answered:
the plural and the article each broke the span. The store's alias relic
(`aliases.jsonl`, surface to subject, read when `surfaces.ehmap` is
emitted) is how a store says which wordings name a subject, and the
encoder knows its own: it writes `out/aliases.jsonl` with the plural of
each class (`people` for person, `trucks`, `tanks`) and, for each pair, the
plural of either side and an article before the second (`people near a
tank`, `person near the tank`, `people near tanks`); a stencil takes
neither. `publish.sh` hands the file to the ingest door as
`subject_aliases`, which copies it into the store before the relics.
Measured on a copy of the store before the plumbing existed: with those
wordings bound, "find all people near a tank" answers frames 244 to 260
and "find all people near a truck" its 71 frames; "which frames show
people near a tank" still declines (the verb `show` is not a `shown in`
reading).

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

## 6. Publish to the appliance: the pool, its sidecars, five-frame playback

```bash
./publish.sh eworld            # --replace to re-encode an existing pool of that name
```

`publish.sh` is the clip's way into the running appliance (`/opt/mpl`,
or `--home`). It does four things, in this order, and prints each:

1. **Ingest through the front door.** `POST /v1/ingest {source:
   out/triples.jsonl, store: eworld, aliases: ["camera feed"],
   subject_aliases: out/aliases.jsonl}` on the appliance is HaloCore's
   `ehc::ops::ingest`, the one call `halocore ingest` makes. It encodes the
   triples into the native store `<home>/data/stores/eworld/` (the
   appliance's ingest root), emits the serving relics as every ingest does
   (`subjects.tsv`, `surfaces.ehmap`, `sim_subjects.ehsim`,
   `sim_relations.ehsim`, `relation_surfaces.json`, `subj_index.bin`, the
   relation learner's files) and mounts the store as the pool `eworld`.
   `aliases` becomes the store's `scope.json`, the names it answers to in
   a question's scope prefix (below); `subject_aliases` is copied in as
   the store's `aliases.jsonl` before the relics, so the surface map binds
   the encoder's wordings to its subjects. The appliance's `start.sh`
   walks the ingest root at boot, so the pool comes back after a restart
   like every store under `MOE/`.
2. **The preview and its lookups beside the shards.** `<store>/media/`
   gets `preview.mp4`, `preview.index.json`, `frames.lsh` (the keyframe
   vectors), `objects.jsonl` (the object index), `overlay.jsonl`,
   `frames.jsonl`, `videos.jsonl`, `config.json`, `encode_summary.json`
   and the `triples.jsonl` that was ingested. The appliance serves them at
   `GET /v1/pools/eworld/files/media/<name>` (a mounted store's own
   directory only; media, JSON and text only; HTTP Range for the video),
   the same way a document store's `figures/` sit beside its shards.
3. **Relics and the mount, checked.** `halocore relics --index <store>
   --contract` is run and the relic files listed with their sizes; the
   registry (`GET /v1/lakes`) is read back for the pool; the index and a
   Range of the video are fetched through the appliance.
4. **One question** through the query door (`--ask`, default "What is the
   story setting of frame 26?").

**Five-frame playback on the page.** When an answer's facts name a frame
(`frame 26` as subject or object, or among the objects also seen) of a
store that carries `media/preview.index.json`, the answer card shows that
store's preview and plays five sampled frames from the one named: the
frame itself and the four survivors after it, each for its own slot on the
source clock, then pauses. Chips switch between the frames an answer
names; the strip under the video lists the five frames (time, scene,
objects and words) and clicking one seeks to it. This is
`frontend/frame_playback.js` (L3M `halocore_v1`), one GET per store per
page; a store without the sidecar changes nothing on the page.

**The scope prefix: `camera feed: find all people near a tank`.** The
appliance mounts seventy-odd pools, and a question the clip cannot answer
used to fall past the lake to the passage reader of some other corpus
("people near fuel" returned a line from a graph-theory book). A label
before a colon and a space names the stores a question is for: a pool's
name (`eworld:`), or an alias the store declares in its `scope.json`
(`camera feed:`, which `publish.sh` gives every pool it publishes, plus
any `--alias`). HaloCore's query door (`ehc/sim/ask.hpp`) resolves it and
runs every stage over those stores only, including the record lanes, the
passage reader and the appliance's document pools; the reply carries
`scope: {label, stores}`, and a decline says `in camera feed: ...`. The
question after the colon is unchanged, so everything that works without
the prefix works with it. A label no store answers to is left in the
question; `site:AUH-00001` (no space after the colon) is a record id. The
same prefix works on the CLI: `halocore query --text "camera feed: what
is near the tank?"` over the same mounts. Measured through `/v1/chat`
after the install: the three questions ("find all people near a tank",
"near a truck", "near fuel") answer frames 244 to 260, the 71 truck
frames, and a clean decline inside the camera feed; "what is near the
tank?" stops going to the wiki pool's thesaurus ids and answers from the
clip.

## OpenCV as its own detector

A Haar or DNN detector is the pattern in `how_to_use_video_encoding.md`: a
script that prints `yolo_json` and is passed to `ehc_video_encode --detector`.
That path fills `tags` and drops `xyxy`. Use it when you only need the
concept query. Use `aerial_encode` when the reel has to show rectangles.
