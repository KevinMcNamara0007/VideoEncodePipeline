#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Write pass-2 box coordinates. No stills.

Pass 2 kept a class name per detection and dropped the rectangle. This
reads the same film at 2 frames a second, runs YOLOv8 and then
YOLOv8-World, and writes normalized boxes. Two boxes of one class stay
when they do not overlap. Florence is not called.

    python3 pass2_boxes.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "input" / "range_pass.mp4"
OUT = ROOT / "out" / "pass2" / "boxes.jsonl"
COCO = ROOT / "models" / "yolov8n.pt"
WORLD_WEIGHTS = ROOT / "models" / "yolov8s-worldv2.pt"
WORLD_CLASSES = [
    "rifle", "machine gun", "pistol",
    "tank", "humvee", "helicopter", "artillery",
]
CONF = 0.10
BATCH = 8
SAMPLE_FPS = 2


def iou(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    if union <= 0:
        return 0.0
    return inter / union


def keep(existing, incoming):
    """Keep a second box of the same class when it is a different object."""
    for box in incoming:
        replaced = False
        for prev in existing:
            if prev["class"] != box["class"]:
                continue
            if iou(prev, box) < 0.45:
                continue
            if box["conf"] > prev["conf"]:
                prev.update(box)
            replaced = True
            break
        if not replaced:
            existing.append(box)


def boxes_of(result, source):
    found = []
    if result.boxes is None or len(result.boxes) == 0:
        return found
    height, width = result.orig_shape
    names = result.names
    for cls_id, conf, xy in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        x1, y1, x2, y2 = [float(v) for v in xy]
        found.append({
            "class": str(names[int(cls_id)]),
            "conf": round(float(conf), 3),
            "source": source,
            "x": round(min(1.0, max(0.0, x1 / width)), 4),
            "y": round(min(1.0, max(0.0, y1 / height)), 4),
            "w": round(min(1.0, max(0.0, (x2 - x1) / width)), 4),
            "h": round(min(1.0, max(0.0, (y2 - y1) / height)), 4),
        })
    return found


def frames_of(video, width, height):
    """Sample at 2 fps, stretched the way pass 2 saw the film.

    The stretch is linear, so a box as a fraction of this frame is the
    same fraction of the original picture. The frames are not written.
    """
    nbytes = width * height * 3
    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
            "-vf", "fps=%d,scale=%d:%d" % (SAMPLE_FPS, width, height),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    batch = []
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            batch.append(image.copy())
            if len(batch) >= BATCH:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def detect(weights, source, setup=None):
    from ultralytics import YOLO
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(weights))
    if setup:
        setup(model)
    # Pass 2 classified the stretched square, not the 16:9 frame.
    width, height = 640, 640
    found = {}
    seen = 0
    for batch in frames_of(VIDEO, width, height):
        results = model.predict(
            batch, conf=CONF, imgsz=640, device=device, verbose=False,
        )
        for result in results:
            found[seen] = boxes_of(result, source)
            seen += 1
        if seen % 80 < BATCH:
            print("[boxes] %s %d" % (source, seen), flush=True)
    del model
    print("[boxes] %s frames %d on %s" % (source, seen, device), flush=True)
    return found


def main():
    if not VIDEO.is_file():
        raise SystemExit("film not found: %s" % VIDEO)
    coco = detect(COCO, "coco")
    world = detect(WORLD_WEIGHTS, "world", lambda model: model.set_classes(WORLD_CLASSES))
    count = max(len(coco), len(world))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    kept = 0
    for idx in range(count):
        boxes = list(coco.get(idx) or [])
        keep(boxes, world.get(idx) or [])
        boxes.sort(key=lambda b: -b["conf"])
        boxes = boxes[:16]
        kept += len(boxes)
        lines.append(json.dumps({
            "frame_idx": idx,
            "t": round(idx / float(SAMPLE_FPS), 3),
            "boxes": boxes,
        }))
    tmp = OUT.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(OUT)
    print("[boxes] wrote %s (%d frames, %d boxes)" % (OUT, count, kept), flush=True)


if __name__ == "__main__":
    main()
