#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""One-time weights conversion: Ultralytics .pt -> .onnx for the C++ pipeline.

The pipeline itself is C++ (OpenCV DNN reads the .onnx). This is the only
Python left, and it runs once per checkpoint, not per clip. It writes
<model>.onnx and <model>.names.json (the class order of the output head).

    python3 tools/export_onnx.py models/yolov8n.pt
    python3 tools/export_onnx.py models/yolov8s-worldv2.pt \\
        --classes "rifle,machine gun,pistol" --out models/yolov8s-worldv2-weapons.onnx

YOLO-World bakes the --classes text embeddings into the graph, so one .onnx
answers one prompt list. Needs ultralytics, onnx and (for World) CLIP.
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--classes", default="",
                    help="comma list; YOLO-World only (set_classes before export)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    if args.classes:
        model.set_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    made = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12,
                             simplify=True, dynamic=False))
    out = args.out or made
    if out.resolve() != made.resolve():
        shutil.move(str(made), str(out))
    names = model.names
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    names_path = out.with_suffix(".names.json")
    names_path.write_text(json.dumps(ordered, indent=1) + "\n")
    print(f"[export] {out}  ({len(ordered)} classes -> {names_path.name})")


if __name__ == "__main__":
    main()
