#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Send the pass-2 YOLO index through the HaloCore front door.

YOLOv8 and YOLOv8-World write out/pass2/ocr.json. This step does not run
the models again. It turns that index into triples and posts them to the
native appliance:

    POST /v1/ingest  {source, store}

The door encodes the file, writes the lake under HALOAGENT_INGEST_ROOT,
and mounts it. The store argument is a name, not a path.

Subjects are names (``frame 26``, ``truck``, ``armor``, ``range pass``).
A colon id such as ``frame:range_pass:26`` makes the store record-keyed,
and a record-keyed store does not get surfaces.ehmap. Relations are
slugs the relation-surface lexicon already has forms for (depicts,
depicted_by, narrative_location, subtitle, ...). A relation outside
that lexicon is dropped, and relation_surfaces.json is not written.

    python3 pass2_frontdoor.py
    python3 pass2_frontdoor.py --store eworld --ocr out/pass2/ocr.json
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from eworld_surfaces import publish_surfaces


ROOT = Path(__file__).resolve().parent
DEFAULT_OCR = ROOT / "out" / "pass2" / "ocr.json"
DEFAULT_TRIPLES = ROOT / "out" / "pass2" / "triples.jsonl"
CONFIG_ENV = Path("/opt/config/config.env")


def native_port() -> int:
    if not CONFIG_ENV.is_file():
        return 8099
    for line in CONFIG_ENV.read_text().splitlines():
        if line.startswith("HALOAGENT_NATIVE_PORT="):
            value = line.split("=", 1)[1].split("#", 1)[0].strip()
            return int(value or "8099")
    return 8099


def write_triples(ocr_path: Path, dest: Path) -> int:
    """Name subjects and lexicon relations, so the door can emit relics.

    ``frame N`` depicts a class. The class is depicted_by that frame.
    The scene name is the frame's narrative_location, and that scene is
    located at the frame. OCR-index rows are subtitle (on-screen text).
    A later view of an open track is said_to_be_the_same_as the earlier
    frame. A frame with no box is a facet_of no-box.
    """
    rows = [json.loads(line) for line in ocr_path.read_text().splitlines() if line.strip()]
    count = 0

    def emit(fh, subject, relation, obj, qualifier=""):
        nonlocal count
        fh.write(json.dumps({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "qualifier": qualifier,
        }, ensure_ascii=False) + "\n")
        count += 1

    def frame_name(idx):
        return "frame %d" % int(idx)

    def scene_name(raw):
        text = (raw or "").strip()
        if text.startswith("scene:"):
            text = text.split(":", 1)[1]
        return text.replace(":", " ")

    last_ms = str(max(int(row["ts_ms"]) for row in rows)) if rows else "0"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        emit(fh, "range pass", "title", "range pass")
        emit(fh, "range pass", "described_by_source", "yolov8+yolov8-world")
        emit(fh, "range pass", "duration", last_ms + " ms")
        emit(fh, "range pass", "facet_of", "pass 2")
        for row in rows:
            frame = frame_name(row["frame_idx"])
            ms = str(int(row["ts_ms"]))
            track = row.get("track")
            when = ms if track is None else "%s track %s" % (ms, track)
            emit(fh, frame, "part_of", "range pass", ms)
            emit(fh, frame, "point_in_time", ms, "" if track is None else "track %s" % track)
            scene = scene_name(row.get("scene"))
            if scene:
                emit(fh, frame, "narrative_location", scene, ms)
                emit(fh, scene, "location", frame, ms)
            for obj in row.get("objects") or []:
                label = str(obj["class"]).replace(":", " ")
                emit(fh, frame, "depicts", label, when)
                emit(fh, label, "depicted_by", frame, when)
            if row.get("role") == "ocr":
                emit(fh, frame, "subtitle", row.get("reason") or "ocr", ms)
            if row.get("ref_frame") is not None:
                emit(fh, frame, "said_to_be_the_same_as", frame_name(row["ref_frame"]), ms)
            elif row.get("role") != "ocr":
                emit(fh, frame, "facet_of", "no-box", ms)
    return count


def post_ingest(port: int, source: Path, store: str) -> dict:
    body = json.dumps({
        "source": str(source),
        "store": store,
        "append": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/ingest" % port,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("front door ingest failed (%s): %s" % (exc.code, detail))
    return json.loads(raw) if raw else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocr", type=Path, default=DEFAULT_OCR)
    ap.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    ap.add_argument("--store", default="eworld")
    args = ap.parse_args()
    if not args.ocr.is_file():
        raise SystemExit("YOLO index not found: %s" % args.ocr)
    n = write_triples(args.ocr, args.triples)
    port = native_port()
    print("[eworld] %d triples -> %s" % (n, args.triples), flush=True)
    print("[eworld] POST /v1/ingest store=%s port=%d" % (args.store, port), flush=True)
    result = post_ingest(port, args.triples.resolve(), args.store)
    print(json.dumps(result, indent=2))
    store_dir = Path(result.get("path") or "")
    if not store_dir.is_dir():
        raise SystemExit("ingest did not report a store directory")
    # The door compiles surfaces before aliases.jsonl exists. Publish the
    # a/an/the spelling of every subject, then remount that map.
    n_alias = publish_surfaces(store_dir, port)
    print("[eworld] %d determiner surfaces" % n_alias, flush=True)


if __name__ == "__main__":
    main()
