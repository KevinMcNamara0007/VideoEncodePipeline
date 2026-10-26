#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Encode aerial clips with YOLOv8 labels, Florence-2 gaps, and OCR fused
into the EHC streaming video encoder, and write the boxes the vector does
not store.

See recipe_aerial_overlay.md in this directory.

    python3 aerial_overlay_encode.py <clips_dir> <out_dir> \\
        --fps 2 --frame 640x640 --model yolov8n.pt --ocr tesseract
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Before ultralytics imports torch. Florence-2 generate needs the CPU
# fallback for aten::isin on MPS. See florence_gap.py.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

# This file lives in EWORLD. The ehc module is built in the sibling EHC tree.
EHC_ROOT = Path(__file__).resolve().parent.parent / "EHC"


def _import_ehc():
    candidates = []
    env = os.environ.get("EHC_PYTHON_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(EHC_ROOT / "build-py" / "bindings" / "python")
    candidates.append(EHC_ROOT / "build" / "bindings" / "python")
    last = None
    for c in candidates:
        if not c.is_dir():
            continue
        sys.path.insert(0, str(c))
        try:
            import ehc  # noqa: WPS433
            return ehc
        except ImportError as exc:
            last = exc
            sys.path.pop(0)
    raise SystemExit(
        "cannot import ehc. Build the bindings with the same interpreter\n"
        "you are running now:\n"
        f"  cmake -S {EHC_ROOT} -B {EHC_ROOT / 'build-py'} "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DEHC_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)\n"
        f"  cmake --build {EHC_ROOT / 'build-py'} --target ehc_python\n"
        f"last error: {last}"
    )


ehc = _import_ehc()

DIM = 16384
K = 128
NUM_TABLES = 8
HASH_SIZE = 16
SEED = 2300
TOKEN_SEED = 2400  # StreamingVideoConfig::detector_token_seed default
MAX_OCR_TAGS = 8
OCR_MIN_CONF = 40.0  # Tesseract word confidence, 0–100. Stencils often land in the 40s.

ARMOR_CLASSES = {"tank", "apc", "humvee", "artillery"}
GROUND_CLASSES = {"car", "truck", "bus", "motorcycle", "crate", "bridge"}
AIR_NAMES = {"airplane", "helicopter"}
WEAPON_RAW = {"rifle", "machine gun", "pistol", "gun"}
# Second YOLOv8 head (YOLO-World). COCO stays the object head.
WORLD_CLASSES = [
    "rifle", "machine gun", "pistol",
    "tank", "humvee", "helicopter", "artillery",
]


def ffprobe(path: str) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        stderr=subprocess.DEVNULL,
    )
    return json.loads(out)


def video_meta_from_ffprobe(probe: dict, source_path: str) -> dict:
    fps = 0.0
    codec = container = ""
    duration_ms = 0.0
    total_frames = 0
    src_w = src_h = 0
    for s in probe.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        codec = s.get("codec_name", "")
        src_w = int(s.get("width") or 0)
        src_h = int(s.get("height") or 0)
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = s.get(key, "0/0")
            try:
                num, den = val.split("/")
                num, den = float(num), float(den)
                if den > 0 and num > 0:
                    fps = num / den
                    break
            except ValueError:
                continue
        try:
            total_frames = int(s.get("nb_frames") or 0)
        except ValueError:
            total_frames = 0
        break
    fmt = probe.get("format", {})
    container = fmt.get("format_name", "")
    try:
        duration_ms = float(fmt.get("duration") or 0.0) * 1000.0
    except ValueError:
        duration_ms = 0.0
    return {
        "source_path": source_path,
        "fps": fps if fps > 0 else 30.0,
        "duration_ms": duration_ms,
        "total_frames": total_frames,
        "codec": codec,
        "container": container,
        "src_w": src_w,
        "src_h": src_h,
    }


def stream_rgb_frames(path: str, target_fps: float, w: int, h: int):
    """Yield (frame_idx, pts_ms, HxWx3 uint8). pts_ms is frame_idx / target_fps."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
        "-vf", f"fps={target_fps},scale={w}:{h}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            yield idx, (idx * 1000.0) / target_fps, img
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=60)


def ocr_token(text: str):
    chars = [ch for ch in text.upper() if ch.isalnum()]
    if not 2 <= len(chars) <= 16:
        return None
    return "ocr:" + "".join(chars)


def index_token(raw: str, kind: str) -> str:
    """COCO keeps its class string. The world head maps weapons to weapon:*."""
    name = str(raw).strip()
    key = " ".join(name.lower().split())
    if kind == "world" and key in WEAPON_RAW:
        slug = "rifle" if key == "gun" else key.replace(" ", "-")
        return "weapon:" + slug
    if kind == "world":
        return key
    return name


def _area(xyxy) -> float:
    if not xyxy or len(xyxy) != 4:
        return 0.0
    return max(0.0, float(xyxy[2]) - float(xyxy[0])) * max(0.0, float(xyxy[3]) - float(xyxy[1]))


def bare_name(label: str) -> str:
    if label.startswith("weapon:"):
        return label.split(":", 1)[1]
    return label.lower()


def scene_tag(class_names: set) -> str:
    """One rule tag from the class set. Empty when the frame has no boxes."""
    names = set()
    for label in class_names:
        if not label or label.startswith("ocr:"):
            continue
        names.add(bare_name(label))
    if not names:
        return ""
    air = bool(names & AIR_NAMES)
    water = "boat" in names
    ground = bool(names & (GROUND_CLASSES | ARMOR_CLASSES))
    person = "person" in names
    if names & ARMOR_CLASSES:
        return "scene:armor"
    if air and not ground:
        return "scene:air"
    if water and not ground and not air:
        return "scene:water"
    if ground and person:
        return "scene:ground-with-person"
    if ground or names & {"rifle", "machine-gun", "pistol"}:
        return "scene:ground"
    if person:
        return "scene:person"
    return "scene:other"


def tesseract_tsv(png_bytes: bytes, psm: str) -> list:
    """Word rows from `tesseract stdin stdout tsv`. Each row is a dict."""
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", psm, "-l", "eng", "tsv"],
        input=png_bytes,
        capture_output=True,
    )
    if proc.returncode != 0 and not proc.stdout:
        return []
    rows = []
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        if rec.get("level") != "5":
            continue
        try:
            conf = float(rec.get("conf", "-1"))
        except ValueError:
            continue
        text = rec.get("text") or ""
        if conf < OCR_MIN_CONF or not text.strip():
            continue
        try:
            left = int(float(rec["left"]))
            top = int(float(rec["top"]))
            width = int(float(rec["width"]))
            height = int(float(rec["height"]))
        except (KeyError, ValueError):
            continue
        rows.append({
            "text": text.strip(),
            "conf": conf,
            "xyxy": [left, top, left + width, top + height],
        })
    return rows


def png_bytes(bgr) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return b""
    return buf.tobytes()


def build_ocr_schedule(frames_jsonl: Path) -> list:
    """First frame of each distinct stencil is transcribed. The rest are marks.

    The signature is the pass-0 `ocr:` tags after the prompt echo and the
    two-letter specks are removed. A later frame with the same signature
    points at that first frame and is not sent to Florence.
    """
    from halocore_florence import keep_stencil
    rows = [
        json.loads(line)
        for line in frames_jsonl.read_text().splitlines()
        if line.strip()
    ]
    first = {}
    schedule = []
    for ordinal, row in enumerate(rows):
        words = []
        for tag in row.get("tags") or []:
            if not str(tag).startswith("ocr:"):
                continue
            token = str(tag)[4:].upper()
            if not keep_stencil(token) or token in words:
                continue
            words.append(token)
        sig = tuple(sorted(words))
        if not sig:
            schedule.append({
                "role": "mark",
                "reason": "no-text",
                "ref": None,
                "frame_idx": row.get("frame_idx"),
            })
            continue
        if sig in first:
            schedule.append({
                "role": "mark",
                "reason": "redundant",
                "ref": first[sig],
                "frame_idx": row.get("frame_idx"),
            })
            continue
        first[sig] = ordinal
        schedule.append({
            "role": "transcribed",
            "reason": "new-text",
            "ref": None,
            "frame_idx": row.get("frame_idx"),
        })
    return schedule


class AerialDetector(ehc.FrameDetector):
    """YOLO boxes plus OCR strings. Geometry is stashed on `self.records`.

    `detect()` returns labels only. That is what the encoder superposes.
    `records[i]` lines up with survivor i and is what the driver writes to
    overlay.jsonl. The encoder deletes the temp PNGs after `detect()` returns,
    so the stash has to be taken before the next video.
    """

    def __init__(self, model_path: str, conf: float, do_ocr: bool, imgsz: int,
                 weapon_model: str = "", florence_mode: str = "caption",
                 ocr_schedule=None):
        super().__init__()
        from ultralytics import YOLO
        import torch
        self.conf = conf
        self.do_ocr = do_ocr
        self.imgsz = imgsz
        # off | ocr | caption. OCR reads stencil text. Caption also adds
        # class names the prose mentions.
        self.florence_mode = florence_mode
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = YOLO(model_path)
        self.weapon = None
        if weapon_model:
            self.weapon = YOLO(weapon_model)
            # Caption mode asks the world head for weapons only. It labels
            # cargo trucks as humvees, and the caption supplies armor names.
            # OCR-only mode asks for the full military list so YOLO itself
            # is the object and scene source.
            if florence_mode == "caption":
                classes = ["rifle", "machine gun", "pistol"]
            else:
                classes = WORLD_CLASSES
            self.weapon.set_classes(classes)
        self.florence = None
        if florence_mode != "off":
            from halocore_florence import HaloFlorence
            role = "OCR only" if florence_mode == "ocr" else "caption + OCR"
            print(f"[aerial] Florence-2 is the HaloCore door ({role})", flush=True)
            self.florence = HaloFlorence()
        # Pass 1: one entry per pass-0 survivor. Florence runs only where
        # role == "transcribed". Every other frame is a sidecar mark.
        self.ocr_schedule = ocr_schedule
        self.records = []

    def name(self):
        return "aerial_yolo_florence_ocr"

    @staticmethod
    def _index_results(results):
        by_name = {}
        for r in results:
            by_name[Path(getattr(r, "path", "")).name] = r
        return by_name

    @staticmethod
    def _take_yolo(boxes, result, kind):
        if result is None or result.boxes is None or not len(result.boxes):
            return
        names = result.names
        xyxy = result.boxes.xyxy.tolist()
        for cls_id, conf, xy in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist(), xyxy
        ):
            label = index_token(names[int(cls_id)], kind)
            incoming = {
                "class": label,
                "confidence": float(conf),
                "xyxy": [float(v) for v in xy],
                "ocr": [],
                "source": kind,
            }
            kept = False
            for prev in boxes:
                if prev["class"] != label or prev["class"].startswith("ocr:"):
                    continue
                if incoming["confidence"] > prev["confidence"]:
                    prev["confidence"] = incoming["confidence"]
                    prev["xyxy"] = incoming["xyxy"]
                    prev["source"] = kind
                kept = True
                break
            if not kept:
                boxes.append(incoming)

    def detect(self, image_paths):
        import cv2
        self.records = []
        out = []
        if not image_paths:
            return out
        try:
            results = self.model.predict(
                source=list(image_paths),
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                save=False,
                stream=False,
            )
        except Exception as exc:  # detector contract: do not raise
            sys.stderr.write(f"[aerial] yolo failed: {exc}\n")
            self.records = [{"boxes": [], "scene": ""} for _ in image_paths]
            return [ehc.FrameDetections() for _ in image_paths]

        by_name = self._index_results(results)
        captions = {}
        schedule = self.ocr_schedule
        use_schedule = bool(schedule) and len(schedule) == len(image_paths)
        if schedule and not use_schedule:
            print(
                f"[aerial] ocr schedule has {len(schedule)} row(s) for "
                f"{len(image_paths)} survivor(s); sending every frame",
                flush=True,
            )
        if self.florence is not None:
            from halocore_florence import figure_words, phrases_in
            send = list(image_paths)
            if use_schedule:
                send = [
                    p for p, slot in zip(image_paths, schedule)
                    if slot["role"] == "transcribed"
                ]
                print(
                    f"[aerial] florence ocr {len(send)}/{len(image_paths)} "
                    f"frame(s); {len(image_paths) - len(send)} marked in the sidecar",
                    flush=True,
                )
            captions = self.florence.read_many(send)
        else:
            figure_words = phrases_in = None
        world_by_name = {}
        if self.weapon is not None:
            try:
                world_results = self.weapon.predict(
                    source=list(image_paths),
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    save=False,
                    stream=False,
                )
                world_by_name = self._index_results(world_results)
            except Exception as exc:
                sys.stderr.write(f"[aerial] weapon yolo failed: {exc}\n")

        for i, path in enumerate(image_paths):
            fd = ehc.FrameDetections()
            rec = {"boxes": [], "scene": ""}
            boxes = []
            self._take_yolo(boxes, by_name.get(Path(path).name), "coco")
            self._take_yolo(boxes, world_by_name.get(Path(path).name), "world")

            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            caption = captions.get(Path(path).stem, "")
            # COCO calls a canvas cargo truck a train. Drop that only when
            # the caption is allowed to veto YOLO. Pass 0 keeps the YOLO label.
            if (self.florence_mode == "caption" and caption
                    and not any(w.startswith("train") for w in caption.lower().split())):
                boxes = [b for b in boxes if b["class"] != "train"]
            if caption and self.florence_mode == "caption" and phrases_in is not None:
                already = {b["class"] for b in boxes}
                for token in phrases_in(caption):
                    if token in already:
                        continue
                    # The caption names the thing. It does not return a box.
                    boxes.append({
                        "class": token,
                        "confidence": 0.5,
                        "xyxy": [0.0, 0.0, 0.0, 0.0],
                        "ocr": [],
                        "source": "halocore",
                        "overlay": False,
                    })
                    already.add(token)
            words = figure_words(caption) if caption and figure_words is not None else []
            side = None
            if use_schedule:
                slot = schedule[i]
                if slot["role"] != "transcribed":
                    # The plate was already read on an earlier frame.
                    # This frame keeps objects and scene, not a second copy
                    # of the stencil.
                    words = []
                side = {
                    "role": slot["role"],
                    "reason": slot["reason"],
                    "ref_ordinal": slot["ref"],
                }
            if words:
                hosts = [
                    b for b in boxes
                    if b.get("overlay", True) and not str(b["class"]).startswith("ocr:")
                ]
                host = max(hosts, key=lambda b: _area(b["xyxy"])) if hosts else None
                # A speck (a distant person) should not inherit a runway mark.
                if host is not None and words and _area(host["xyxy"]) >= 48 * 48:
                    for word in words:
                        if word not in host["ocr"]:
                            host["ocr"].append(word)
                else:
                    for word in words:
                        boxes.append({
                            "class": "ocr:" + word,
                            "confidence": 0.5,
                            "xyxy": [12.0, 48.0, 12.0, 48.0],
                            "ocr": [word],
                            "source": "halocore-ocr",
                        })
            if self.do_ocr and bgr is not None and boxes:
                h, w = bgr.shape[:2]
                ocr_left = MAX_OCR_TAGS
                for b in boxes:
                    if ocr_left <= 0:
                        break
                    x0, y0, x1, y1 = b["xyxy"]
                    x0 = max(0, int(x0))
                    y0 = max(0, int(y0))
                    x1 = min(w, int(x1))
                    y1 = min(h, int(y1))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    if str(b["class"]).startswith("ocr:"):
                        continue
                    crop = bgr[y0:y1, x0:x1]
                    ch, cw = crop.shape[:2]
                    if ch < 32 or cw < 32:
                        crop = cv2.resize(
                            crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                        )
                    seen = set()
                    for word in tesseract_tsv(png_bytes(crop), "7"):
                        tok = ocr_token(word["text"])
                        if tok and tok not in seen:
                            seen.add(tok)
                            b["ocr"].append(tok[4:])
                            ocr_left -= 1
                            if ocr_left <= 0:
                                break
            elif self.do_ocr and bgr is not None and not any(
                str(b["class"]).startswith("ocr:") for b in boxes
            ):
                # psm 11 is the sparse page. It returns nothing on a single
                # stencil, so fall through to a uniform block (psm 6).
                words = tesseract_tsv(png_bytes(bgr), "11")
                if not words:
                    words = tesseract_tsv(png_bytes(bgr), "6")
                for word in words:
                    tok = ocr_token(word["text"])
                    if not tok:
                        continue
                    # A full frame with no detector box is mostly dirt and sky.
                    # Keep a stencil (a digit, or a word of 4+ letters) and
                    # drop the two-letter noise Tesseract emits on dust.
                    body = tok[4:]
                    if not (any(ch.isdigit() for ch in body) or len(body) >= 4):
                        continue
                    boxes.append({
                        "class": tok,
                        "confidence": word["conf"] / 100.0,
                        "xyxy": [float(v) for v in word["xyxy"]],
                        "ocr": [body],
                        "source": "tesseract",
                    })
                    if len(boxes) >= MAX_OCR_TAGS:
                        break

            best = {}
            for b in boxes:
                label = b["class"]
                prev = best.get(label)
                if prev is None or b["confidence"] > prev:
                    best[label] = b["confidence"]
                for text in b["ocr"]:
                    tok = "ocr:" + text
                    prev = best.get(tok)
                    conf = max(b["confidence"], 0.5)
                    if prev is None or conf > prev:
                        best[tok] = conf
            scene = scene_tag({b["class"] for b in boxes if not b["class"].startswith("ocr:")})
            if scene:
                best.setdefault(scene, 0.55)
            # Assign the list in one set. nanobind's def_rw on a vector
            # returns a copy, so append() never reaches the C++ object.
            ocr_added = 0
            tags = []
            for label, conf in sorted(best.items(), key=lambda kv: -kv[1]):
                if label.startswith("ocr:"):
                    if ocr_added >= MAX_OCR_TAGS:
                        continue
                    ocr_added += 1
                tag = ehc.DetectedTag()
                tag.label = label
                tag.confidence = float(conf)
                tags.append(tag)
            fd.tags = tags
            rec["boxes"] = boxes
            rec["scene"] = scene
            rec["ocr_sidecar"] = side
            self.records.append(rec)
            out.append(fd)
        return out


def encode_one(video_id, src, cfg, target_fps, frame_w, frame_h):
    t0 = time.time()
    probe = ffprobe(src)
    vmeta = video_meta_from_ffprobe(probe, src)
    vmeta["video_id"] = video_id
    enc = ehc.StreamingVideoEncoder(cfg)
    enc.begin_video(video_id=video_id, fps=vmeta["fps"])
    n_input = 0
    for idx, pts_ms, img in stream_rgb_frames(src, target_fps, frame_w, frame_h):
        enc.on_frame_rgb(image=img, pts_ms=pts_ms, frame_idx=idx)
        n_input += 1
    encoded = enc.finalize()
    s = enc.stats
    stats = {
        "input_frames": n_input,
        "frames_seen": s.frames_seen,
        "dropped_phash": s.dropped_phash,
        "dropped_vsa": s.dropped_vsa,
        "dropped_stage2": s.dropped_stage2,
        "survivors_kept": len(encoded),
        "detector_tokens_emitted": s.detector_tokens_emitted,
        "detector_tokens_dropped": s.detector_tokens_dropped,
    }
    return vmeta, encoded, stats, (time.time() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exts", default="mp4,webm,mov,mkv,avi,m4v")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--frame", default="640x640")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ocr", choices=("tesseract", "none"), default="tesseract")
    root = Path(__file__).resolve().parent
    ap.add_argument("--weapon-model",
                    default=str(root / "models" / "yolov8s-worldv2.pt"))
    ap.add_argument("--florence", default="halocore",
                    help="halocore = caption + OCR; ocr = stencil text only; none = skip")
    ap.add_argument("--ocr-schedule", default="",
                    help="pass-0 frames.jsonl. Florence transcribes the first "
                         "frame of each distinct stencil; other frames are "
                         "marks in ocr_sidecar.jsonl")
    args = ap.parse_args()

    spec = args.frame.lower().split("x")
    if len(spec) != 2:
        raise SystemExit("--frame must be WxH, for example 640x640")
    frame_w, frame_h = int(spec[0]), int(spec[1])
    if frame_w < 320 or frame_h < 320:
        raise SystemExit("--frame below 320x320 is too small for YOLO and OCR")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exts = {f".{e.lower().strip()}" for e in args.exts.split(",") if e.strip()}
    sources = sorted(
        p for p in args.source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    if args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"no clips under {args.source_dir}")
    print(f"[aerial] {len(sources)} clips  fps={args.fps}  frame={frame_w}x{frame_h}",
          flush=True)
    florence_mode = {
        "": "off", "none": "off", "off": "off",
        "ocr": "ocr",
    }.get(args.florence, "caption")
    if florence_mode == "ocr":
        head = "HaloCore Florence-2 OCR"
    elif florence_mode == "caption":
        head = "HaloCore Florence-2 caption + OCR"
    else:
        head = "no Florence"
    tess = "Tesseract" if args.ocr == "tesseract" else "no Tesseract"
    world = "weapons" if florence_mode == "caption" else "weapons/armor/air"
    print(f"[aerial] heads: yolov8n objects, yolov8s-world {world}, "
          f"{head}, {tess}", flush=True)

    weapon_model = args.weapon_model
    ocr_schedule = None
    if args.ocr_schedule:
        schedule_path = Path(args.ocr_schedule)
        if not schedule_path.is_file():
            raise SystemExit(f"ocr schedule not found: {schedule_path}")
        ocr_schedule = build_ocr_schedule(schedule_path)
        n_send = sum(1 for slot in ocr_schedule if slot["role"] == "transcribed")
        print(
            f"[aerial] ocr schedule {schedule_path.name}: "
            f"{n_send} transcribed, {len(ocr_schedule) - n_send} marked",
            flush=True,
        )
    if weapon_model and not Path(weapon_model).is_file():
        raise SystemExit(f"weapon model not found: {weapon_model}")
    detector = AerialDetector(
        args.model, args.conf, args.ocr == "tesseract",
        imgsz=max(frame_w, frame_h),
        weapon_model=weapon_model,
        florence_mode=florence_mode,
        ocr_schedule=ocr_schedule,
    )
    cfg = ehc.StreamingVideoConfig()
    cfg.dim = DIM
    cfg.k = K
    cfg.frame_w = frame_w
    cfg.frame_h = frame_h
    cfg.seed = SEED
    cfg.color_mode = ehc.ColorMode.rgb
    cfg.dedup_mode = ehc.DedupMode.phash_plus_vsa
    cfg.stage2_pairwise = True
    cfg.max_items_per_frame = 40
    cfg.detector_token_seed = TOKEN_SEED
    cfg.detectors = [detector]

    config_doc = {
        "dim": cfg.dim, "k": cfg.k,
        "frame_w": frame_w, "frame_h": frame_h,
        "seed": cfg.seed,
        "target_fps": args.fps,
        "color_mode": "rgb",
        "dedup_mode": "phash_plus_vsa",
        "stage2_pairwise": True,
        "max_items_per_frame": cfg.max_items_per_frame,
        "lsh_num_tables": NUM_TABLES,
        "lsh_hash_size": HASH_SIZE,
        "detector_cmd": (
            f"aerial_overlay_encode.py --model {args.model} "
            f"--weapon-model {weapon_model} --florence {args.florence} "
            f"--ocr {args.ocr}"
            + (f" --ocr-schedule {args.ocr_schedule}" if args.ocr_schedule else "")
        ),
        "detector_format": f"yolo_json+halocore_{florence_mode}+ocr",
        "detector_token_seed": TOKEN_SEED,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config_doc, indent=2))

    lsh = ehc.BSCLSHIndex(cfg.dim, cfg.k, NUM_TABLES, HASH_SIZE)
    vid_log = open(args.out_dir / "videos.jsonl", "w")
    frame_log = open(args.out_dir / "frames.jsonl", "w")
    overlay_log = open(args.out_dir / "overlay.jsonl", "w")
    sidecar_log = None
    if ocr_schedule is not None:
        sidecar_log = open(args.out_dir / "ocr_sidecar.jsonl", "w")
    enc_log = open(args.out_dir / "encode_log.jsonl", "w")

    batch_vecs, batch_ids = [], []
    next_row = 0
    n_ok = n_fail = 0
    total_survivors = 0
    total_input_frames = 0
    t_start = time.time()

    for vid_id, src in enumerate(sources, start=1):
        try:
            vmeta, encoded, stats, wall_ms = encode_one(
                vid_id, str(src), cfg, args.fps, frame_w, frame_h
            )
        except Exception as exc:
            n_fail += 1
            enc_log.write(json.dumps({
                "video_id": vid_id, "source": str(src), "error": str(exc),
            }) + "\n")
            print(f"[aerial] FAIL #{vid_id} {src.name}: {exc}", flush=True)
            continue

        records = list(detector.records)
        if len(records) != len(encoded):
            # Stage-2 and detect() run on the same survivor list. A mismatch
            # means the stash was not taken. Keep the tags, skip the boxes.
            sys.stderr.write(
                f"[aerial] {src.name}: {len(encoded)} survivors, "
                f"{len(records)} overlay records\n"
            )
            records = []

        vid_log.write(json.dumps(vmeta) + "\n")
        for i, ef in enumerate(encoded):
            row_id = next_row
            next_row += 1
            frame_log.write(json.dumps({
                "video_id": ef.meta.video_id,
                "frame_idx": ef.meta.frame_idx,
                "ts_ms": ef.meta.ts_ms,
                "row_id": row_id,
                "tags": list(ef.tags) if ef.tags else [],
            }) + "\n")
            if records:
                rec = records[i]
                side = rec.get("ocr_sidecar")
                if side is not None and sidecar_log is not None:
                    ref_ord = side.get("ref_ordinal")
                    ref_frame = None
                    if ref_ord is not None and ocr_schedule is not None:
                        ref_frame = ocr_schedule[ref_ord].get("frame_idx")
                    media = f"frame:range_pass:{ef.meta.frame_idx}"
                    if side["role"] == "transcribed":
                        seen = []
                        for box in rec.get("boxes") or []:
                            for word in box.get("ocr") or []:
                                if word not in seen:
                                    seen.append(word)
                            label = str(box.get("class") or "")
                            if label.startswith("ocr:") and label[4:] not in seen:
                                seen.append(label[4:])
                        tags = ["ocr:" + word for word in seen]
                    elif ref_frame is not None:
                        tags = [f"mark:frame:range_pass:{ref_frame}"]
                    else:
                        tags = ["mark:no-text"]
                    sidecar_log.write(json.dumps({
                        "frame_idx": ef.meta.frame_idx,
                        "ts_ms": ef.meta.ts_ms,
                        "media": media,
                        "bridge": "ocr",
                        "role": side["role"],
                        "reason": side["reason"],
                        "ref_frame": ref_frame,
                        "tags": tags,
                    }) + "\n")
                overlay_log.write(json.dumps({
                    "video_id": ef.meta.video_id,
                    "source_path": vmeta["source_path"],
                    "frame_idx": ef.meta.frame_idx,
                    "ts_ms": ef.meta.ts_ms,
                    "row_id": row_id,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "src_w": vmeta["src_w"],
                    "src_h": vmeta["src_h"],
                    "scene": rec.get("scene") or "",
                    "boxes": rec.get("boxes") or [],
                }) + "\n")
            batch_vecs.append(ef.vec)
            batch_ids.append(row_id)

        enc_log.write(json.dumps({
            "video_id": vid_id, "source": src.name, "wall_ms": wall_ms, **stats,
        }) + "\n")
        n_ok += 1
        total_survivors += stats["survivors_kept"]
        total_input_frames += stats["input_frames"]
        print(
            f"[aerial] #{vid_id}/{len(sources)} {src.name[:40]:40s} "
            f"in={stats['input_frames']:5d} kept={stats['survivors_kept']:4d} "
            f"tags={stats['detector_tokens_emitted']} "
            f"dropped={stats['detector_tokens_dropped']} "
            f"wall={wall_ms:8.0f}ms",
            flush=True,
        )

    if batch_vecs:
        lsh.add_items(batch_vecs, batch_ids)
    lsh.save(str(args.out_dir / "frames.lsh"))
    for fh in (vid_log, frame_log, overlay_log, sidecar_log, enc_log):
        if fh is not None:
            fh.close()

    elapsed = time.time() - t_start
    summary = {
        "videos_attempted": len(sources),
        "videos_ok": n_ok,
        "videos_failed": n_fail,
        "total_input_frames": total_input_frames,
        "total_survivors": total_survivors,
        "elapsed_sec": elapsed,
        "lsh_size": lsh.size(),
        "model": args.model,
        "ocr": args.ocr,
    }
    (args.out_dir / "encode_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[aerial] DONE  survivors={total_survivors}  lsh={lsh.size()}", flush=True)


if __name__ == "__main__":
    main()
