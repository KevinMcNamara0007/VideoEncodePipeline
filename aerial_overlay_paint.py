#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Paint overlay.jsonl back onto the source frames and write a preview mp4.

The preview is the survivor keyframes, one after another, at 2 fps. Boxes
are in encoder-frame pixels and are scaled by src_w/frame_w and src_h/frame_h,
matching ffmpeg scale=W:H (stretched, not letterboxed).

    python3 aerial_overlay_paint.py <out_dir> <preview.mp4>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2


def grab_frame(source: str, ts_ms: float):
    """One PNG frame at ts_ms. Seek after -i so the sample clock stays close."""
    sec = max(0.0, ts_ms / 1000.0)
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", source, "-ss", f"{sec:.3f}",
            "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    arr = np_from_png(proc.stdout)
    return arr


def np_from_png(data: bytes):
    import numpy as np
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def color_for(label: str):
    h = abs(hash(label)) % 6
    palette = (
        (60, 220, 80),
        (40, 180, 255),
        (80, 80, 255),
        (200, 180, 40),
        (220, 80, 220),
        (180, 180, 180),
    )
    return palette[h]


def draw_row(img, row):
    src_h, src_w = img.shape[:2]
    fw = float(row.get("frame_w") or src_w)
    fh = float(row.get("frame_h") or src_h)
    sx = src_w / fw if fw else 1.0
    sy = src_h / fh if fh else 1.0
    scene = row.get("scene") or ""
    if scene:
        cv2.rectangle(img, (0, 0), (src_w, 36), (0, 0, 0), -1)
        cv2.putText(img, scene, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (240, 240, 240), 2, cv2.LINE_AA)
    for b in row.get("boxes") or []:
        if b.get("overlay") is False:
            continue
        label = b.get("class") or "?"
        xy = b.get("xyxy") or [0, 0, 0, 0]
        x0 = int(round(xy[0] * sx))
        y0 = int(round(xy[1] * sy))
        x1 = int(round(xy[2] * sx))
        y1 = int(round(xy[3] * sy))
        col = color_for(label)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        ocr = b.get("ocr") or []
        if str(label).startswith("ocr:"):
            text = label[4:]
        else:
            text = label
            if ocr:
                text = f"{label}  {ocr[0]}"
        conf = b.get("confidence")
        if isinstance(conf, float):
            text = f"{text}  {conf:.2f}"
        cv2.putText(img, text, (x0, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("preview", type=Path)
    ap.add_argument("--hold", type=float, default=0.5,
                    help="seconds each survivor stays on screen (default 0.5)")
    args = ap.parse_args()

    overlay_path = args.out_dir / "overlay.jsonl"
    if not overlay_path.is_file():
        raise SystemExit(f"missing {overlay_path}")
    rows = []
    for line in overlay_path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("overlay.jsonl is empty")

    plates = args.out_dir / "preview_frames"
    plates.mkdir(parents=True, exist_ok=True)
    written = []
    for i, row in enumerate(rows):
        img = grab_frame(row["source_path"], float(row.get("ts_ms") or 0))
        if img is None:
            print(f"[paint] skip row {i}: no frame from {row.get('source_path')}",
                  file=sys.stderr)
            continue
        img = draw_row(img, row)
        # yuv420p wants even sizes
        h, w = img.shape[:2]
        if w % 2 or h % 2:
            img = cv2.copyMakeBorder(img, 0, h % 2, 0, w % 2,
                                     cv2.BORDER_REPLICATE)
        path = plates / f"frame_{i:05d}.png"
        cv2.imwrite(str(path), img)
        written.append(path)

    if not written:
        raise SystemExit("no frames painted")

    lst = args.out_dir / "preview_frames.txt"
    # Concat resolves relative paths from the list file's directory.
    with lst.open("w") as fh:
        for p in written:
            fh.write(f"file '{p.resolve()}'\n")
            fh.write(f"duration {args.hold}\n")
        fh.write(f"file '{written[-1].resolve()}'\n")

    args.preview.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(lst),
            "-vsync", "vfr", "-pix_fmt", "yuv420p",
            str(args.preview),
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print(f"[paint] {len(written)} frames -> {args.preview}")


if __name__ == "__main__":
    main()
