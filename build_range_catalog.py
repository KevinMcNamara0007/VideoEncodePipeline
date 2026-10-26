#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Build the range-film catalog the Ask page plays.

out/pass2/ocr.json names the classes in each sampled frame. This groups
each class into passes: a stretch of the film where that class stays in
view. A one-second blink does not split a pass. A pass is not a counted
vehicle. Two of the same class in one shot are one pass, and a vehicle
that leaves and comes back is two.

Plates stay on the frames Florence read. A pass carries a plate when
that plate was read on a frame inside the pass. The letters are not
proven to sit on one box.

out/pass2/boxes.jsonl holds the rectangles, normalized to the film.
The page draws those on the one film. This script stores no stills and
no separate clips.

    python3 build_range_catalog.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "out" / "pass2" / "ocr.json"
READS = ROOT / "out" / "pass2" / "ocr_reads.jsonl"
BOXES = ROOT / "out" / "pass2" / "boxes.jsonl"
VIDEO = ROOT / "input" / "range_pass.mp4"
OUT = Path("/opt/mpl/frontend/range")
SAMPLE_FPS = 2
# Missed samples shorter than this stay in the same pass (2 frames = 1s).
GAP_FRAMES = 2
PAD_S = 0.35

NOUNS = {
    "person": ("person", "people"),
    "tank": ("tank", "tanks"),
    "truck": ("truck", "trucks"),
    "humvee": ("humvee", "humvees"),
    "helicopter": ("helicopter", "helicopters"),
    "airplane": ("airplane", "airplanes"),
    "boat": ("boat", "boats"),
    "train": ("train", "trains"),
    "artillery": ("artillery piece", "artillery"),
    "dining table": ("dining table", "dining tables"),
    "stop sign": ("stop sign", "stop signs"),
    "parking meter": ("parking meter", "parking meters"),
    "traffic light": ("traffic light", "traffic lights"),
}
# The world head labels some whole ground shots "helicopter" at low
# confidence. A box at or above this is an aircraft-sized detection.
HELI_CONF = 0.45
NOTES = {
    "helicopter": "Ground shots with only a low-confidence helicopter label are left out. The same aircraft is sometimes also boxed as an airplane.",
}


def load_rows():
    rows = []
    for line in INDEX.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_reads():
    reads = {}
    if not READS.is_file():
        return reads
    for line in READS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reads[int(row["frame_idx"])] = list(row.get("tokens") or [])
    return reads


def load_boxes():
    boxes = {}
    if not BOXES.is_file():
        return boxes
    for line in BOXES.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kept = []
        for box in row.get("boxes") or []:
            if box.get("class") == "helicopter" and float(box.get("conf") or 0) < HELI_CONF:
                continue
            kept.append(box)
        boxes[str(int(row["frame_idx"]))] = kept
    return boxes


def group_frames(frames):
    frames = sorted(set(frames))
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if frame <= groups[-1][-1] + GAP_FRAMES:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def clock(frame_idx):
    return frame_idx / float(SAMPLE_FPS)


def main():
    if not INDEX.is_file():
        raise SystemExit("index not found: %s" % INDEX)
    if not VIDEO.is_file():
        raise SystemExit("film not found: %s" % VIDEO)
    rows = load_rows()
    reads = load_reads()
    boxes = load_boxes()
    duration = max(clock(int(row["frame_idx"])) for row in rows) + 0.5
    by_class = defaultdict(list)
    classes_at = {}
    for row in rows:
        idx = int(row["frame_idx"])
        names = []
        for obj in row.get("objects") or []:
            name = obj.get("class") or ""
            if not name:
                continue
            # A helicopter pass is only frames with a real helicopter box.
            if name == "helicopter" and boxes and not any(
                b.get("class") == "helicopter" for b in (boxes.get(str(idx)) or [])
            ):
                continue
            names.append(name)
            by_class[name].append(idx)
        classes_at[idx] = names

    OUT.mkdir(parents=True, exist_ok=True)
    link = OUT / "range_pass.mp4"
    if not link.exists():
        os.symlink(VIDEO, link)

    classes_out = {}
    for name in sorted(by_class, key=lambda n: (-len(by_class[n]), n)):
        one, many = NOUNS.get(name, (name, name + "s"))
        visits = []
        for number, frames in enumerate(group_frames(by_class[name]), start=1):
            start = clock(frames[0])
            end = clock(frames[-1]) + 0.5
            plates = []
            also = []
            for frame in frames:
                for token in reads.get(frame) or []:
                    if token not in plates:
                        plates.append(token)
                for other in classes_at.get(frame) or []:
                    if other != name and other not in also:
                        also.append(other)
            ident = "%s-%02d" % (name.replace(" ", "-"), number)
            play_start = max(0.0, start - PAD_S)
            play_end = min(duration, end + PAD_S)
            visits.append({
                "id": ident,
                "start_s": round(play_start, 3),
                "end_s": round(play_end, 3),
                "frames": frames,
                "plates": plates,
                "also": also,
            })
        entry = {"one": one, "many": many, "frames": len(set(by_class[name])), "visits": visits}
        if name in NOTES:
            entry["note"] = NOTES[name]
        classes_out[name] = entry

    plate_frames = defaultdict(list)
    for frame, tokens in reads.items():
        for token in tokens:
            plate_frames[token].append(frame)
    plates_out = {}
    for token in sorted(plate_frames, key=lambda t: (-len(plate_frames[t]), t)):
        visits = []
        for number, frames in enumerate(group_frames(plate_frames[token]), start=1):
            start = clock(frames[0])
            end = clock(frames[-1]) + 0.5
            seen = []
            for frame in frames:
                for other in classes_at.get(frame) or []:
                    if other not in seen:
                        seen.append(other)
            ident = "plate-%s-%02d" % (token, number)
            play_start = max(0.0, start - PAD_S)
            play_end = min(duration, end + PAD_S)
            visits.append({
                "id": ident,
                "start_s": round(play_start, 3),
                "end_s": round(play_end, 3),
                "frames": frames,
                "plates": [token],
                "also": seen,
            })
        plates_out[token] = {"frames": len(set(plate_frames[token])), "visits": visits}

    catalog = {
        "film": "range pass",
        "duration_s": round(duration, 3),
        "video": "/frontend/range/range_pass.mp4",
        "sample_fps": SAMPLE_FPS,
        "blurb": (
            "Each button is one stretch of the range film where that thing stays in view. "
            "A vehicle that leaves and comes back counts again. Two of the same kind in one shot count once, and each one keeps its own box on the picture. "
            "A plate is letters read during that stretch, not proof they are painted on one truck. "
            "Choose a pass and it plays above."
        ),
        "boxes": boxes,
        "classes": classes_out,
        "plates": plates_out,
    }
    tmp = OUT / "catalog.json.tmp"
    tmp.write_text(json.dumps(catalog) + "\n")
    os.replace(tmp, OUT / "catalog.json")
    boxed = sum(len(v) for v in boxes.values())
    print("[range] %d classes, %d plate tokens, %d boxed frames, %d boxes" % (
        len(classes_out), len(plates_out), len(boxes), boxed), flush=True)
    print("[range] wrote %s" % (OUT / "catalog.json"), flush=True)


if __name__ == "__main__":
    main()
