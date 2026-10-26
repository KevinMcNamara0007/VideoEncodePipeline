#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Read the pass-2 OCR index and encode the text through the front door.

out/pass2/ocr.json names the frames YOLOv8 and YOLOv8-World queued for
OCR. This step samples those stills from input/range_pass.mp4 at the same
2 fps, 640x640 clock, reads them with the HaloCore Florence door, and
posts the stencil tokens:

    POST /v1/ingest  {source, store}

This store shares frame names with the detection lake. It does not
assert a subject and relation the detection lake already holds for a
different object: the Ask box would then show one lake and hide the
other. The same object is kept. A frame's title is the plate token.
Each token is on-screen text of that frame, and the token is shown in
the frame.

    python3 pass2_ocr_frontdoor.py
    python3 pass2_ocr_frontdoor.py --store eworld_ocr
    python3 pass2_ocr_frontdoor.py --from-reads
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from eworld_surfaces import contradicts, load_claims, publish_surfaces
from halocore_florence import HaloFlorence, figure_words
from pass2_frontdoor import native_port


ROOT = Path(__file__).resolve().parent
DEFAULT_OCR = ROOT / "out" / "pass2" / "ocr.json"
DEFAULT_VIDEO = ROOT / "input" / "range_pass.mp4"
DEFAULT_READS = ROOT / "out" / "pass2" / "ocr_reads.jsonl"
DEFAULT_TRIPLES = ROOT / "out" / "pass2" / "ocr_triples.jsonl"
DEFAULT_DETECTION = ROOT / "out" / "pass2" / "triples.jsonl"


def candidate_rows(ocr_path: Path) -> list:
    rows = [json.loads(line) for line in ocr_path.read_text().splitlines() if line.strip()]
    picked = [row for row in rows if row.get("role") == "ocr"]
    if not picked:
        raise SystemExit("no role=ocr rows in %s" % ocr_path)
    return picked


def extract_frames(video: Path, rows: list, dest: Path) -> dict:
    """Write frame_NNNNNN.png for each queued sample. Same fps=2, 640 clock."""
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "frame_%06d.png"
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(video),
        "-vf", "fps=2,scale=640:640",
        "-start_number", "0",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    found = {}
    missing = []
    for row in rows:
        idx = int(row["frame_idx"])
        path = dest / ("frame_%06d.png" % idx)
        if path.is_file():
            found[idx] = path
        else:
            missing.append(idx)
    if missing:
        raise SystemExit("missing stills for frame %s" % missing[:8])
    return found


def primary_token(words: list) -> str:
    for word in words:
        if any(ch.isdigit() for ch in word):
            return word
    return words[0] if words else ""


def write_triples(rows: list, reads: dict, dest: Path, claims: dict = None) -> int:
    """Write OCR triples that do not contradict ``claims``.

    ``claims`` is the detection lake: (subject, relation) -> objects.
    A frame subtitle or a frame part_of here is omitted only because the
    detection lake already holds a different object for that pair.
    """
    claims = claims or {}
    count = 0

    def emit(fh, subject, relation, obj, qualifier=""):
        nonlocal count
        if contradicts(claims, subject, relation, obj):
            return
        fh.write(json.dumps({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "qualifier": qualifier,
        }, ensure_ascii=False) + "\n")
        count += 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        emit(fh, "ocr pass", "title", "ocr pass")
        emit(fh, "ocr pass", "described_by_source", "florence-2")
        emit(fh, "ocr pass", "facet_of", "pass 2")
        for row in rows:
            idx = int(row["frame_idx"])
            frame = "frame %d" % idx
            ms = str(int(row["ts_ms"]))
            words = reads.get(idx) or []
            emit(fh, frame, "point_in_time", ms)
            if not words:
                emit(fh, frame, "facet_of", "no-text", ms)
                continue
            emit(fh, frame, "title", primary_token(words), ms)
            for word in words:
                emit(fh, word, "subtitle", frame, ms)
                emit(fh, word, "depicted_by", frame, ms)
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
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--reads", type=Path, default=DEFAULT_READS)
    ap.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    ap.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    ap.add_argument("--store", default="eworld_ocr")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--from-reads", action="store_true",
                    help="Rebuild triples from ocr_reads.jsonl. Do not run Florence.")
    args = ap.parse_args()
    if not args.ocr.is_file():
        raise SystemExit("OCR index not found: %s" % args.ocr)
    rows = candidate_rows(args.ocr)
    print("[eworld] %d OCR frame(s) from %s" % (len(rows), args.ocr), flush=True)

    if args.from_reads:
        if not args.reads.is_file():
            raise SystemExit("reads not found: %s" % args.reads)
        reads = {}
        for line in args.reads.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            reads[int(rec["frame_idx"])] = rec.get("tokens") or []
        missing = [int(row["frame_idx"]) for row in rows if int(row["frame_idx"]) not in reads]
        if missing:
            raise SystemExit("reads missing frame %s" % missing[:8])
        with_text = sum(1 for words in reads.values() if words)
        print("[eworld] reads text on %d/%d frame(s)" % (with_text, len(rows)), flush=True)
    else:
        if not args.video.is_file():
            raise SystemExit("video not found: %s" % args.video)
        work = Path(tempfile.mkdtemp(prefix="eworld_ocr_frames_"))
        try:
            stills = extract_frames(args.video, rows, work)
            print("[eworld] stills %d" % len(stills), flush=True)
            captions = HaloFlorence().read_many([stills[int(row["frame_idx"])] for row in rows])
            reads = {}
            args.reads.parent.mkdir(parents=True, exist_ok=True)
            with args.reads.open("w") as fh:
                for row in rows:
                    idx = int(row["frame_idx"])
                    stem = stills[idx].stem
                    words = figure_words(captions.get(stem, ""))
                    reads[idx] = words
                    fh.write(json.dumps({
                        "frame_idx": idx,
                        "ts_ms": int(row["ts_ms"]),
                        "reason": row.get("reason") or "",
                        "tokens": words,
                    }) + "\n")
            with_text = sum(1 for words in reads.values() if words)
            print("[eworld] florence text on %d/%d frame(s)" % (with_text, len(rows)), flush=True)
        finally:
            if not args.keep_frames:
                shutil.rmtree(work, ignore_errors=True)

    if not args.detection.is_file():
        raise SystemExit("detection triples not found: %s" % args.detection)
    claims = load_claims(args.detection)
    n = write_triples(rows, reads, args.triples, claims)
    port = native_port()
    print("[eworld] %d triples -> %s" % (n, args.triples), flush=True)
    print("[eworld] POST /v1/ingest store=%s port=%d" % (args.store, port), flush=True)
    result = post_ingest(port, args.triples.resolve(), args.store)
    print(json.dumps(result, indent=2))
    store_dir = Path(result.get("path") or "")
    if not store_dir.is_dir():
        raise SystemExit("ingest did not report a store directory")
    n_alias = publish_surfaces(store_dir, port)
    print("[eworld] %d determiner surfaces" % n_alias, flush=True)


if __name__ == "__main__":
    main()
