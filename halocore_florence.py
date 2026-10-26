#!/usr/bin/env python3
"""Florence-2 through the HaloCore document door.

The door (libhalocore_door) runs the vendored Microsoft Florence-2 ONNX
checkpoint: a caption plus <OCR_WITH_REGION>. Python ONNX Runtime cannot
execute that vision graph. This module calls halocore_door_run instead.
"""

import ctypes
import json
import os
import re
import shutil
import tempfile
from ctypes import POINTER, c_char, c_char_p, c_int32, c_uint64, c_uint8
from pathlib import Path

DOOR_LIB = "/opt/bin/libhalocore_door.dylib"
DEFAULT_MODEL = "/opt/mpl/models/florence-2-base-onnx"

# Longer phrases first so "machine gun" wins over a bare "gun".
PHRASES = (
    ("machine gun", "weapon:machine-gun"),
    ("water tower", "water-tower"),
    ("helicopter", "helicopter"),
    ("artillery", "artillery"),
    ("humvee", "humvee"),
    ("rifle", "weapon:rifle"),
    ("pistol", "weapon:pistol"),
    ("tank", "tank"),
    ("crate", "crate"),
    ("bridge", "bridge"),
    ("boat", "boat"),
    ("airplane", "airplane"),
    ("aircraft", "airplane"),
)

_NEG = re.compile(r"\b(no|not|without|n't)\s*$")


class _In(ctypes.Structure):
    _fields_ = [
        ("version", c_int32),
        ("_pad", c_int32),
        ("paths", POINTER(c_char_p)),
        ("count", c_uint64),
        ("outdir", c_char_p),
        ("figures", c_int32),
    ]


class _Out(ctypes.Structure):
    _fields_ = [
        ("code", c_int32),
        ("flag", c_uint8),
        ("_pad", c_uint8 * 3),
        ("message", c_char * 1024),
    ]


def phrases_in(text: str) -> list:
    """Class tokens named in a Florence caption. Skips 'no tank' style denials."""
    low = (text or "").lower()
    found = []
    for phrase, token in PHRASES:
        hit = None
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
            prev = low[max(0, match.start() - 12):match.start()]
            if _NEG.search(prev):
                continue
            hit = token
            break
        if hit and hit not in found:
            found.append(hit)
    # The door says "land vehicle" for a humvee and names the roof gun in
    # the prose. A cargo truck is called a truck, a tank is called a tank.
    if (
        "humvee" not in found
        and "tank" not in found
        and "truck" not in low
        and re.search(r"\bmachine gun\b", low)
        and re.search(r"\bmilitary vehicle\b", low)
    ):
        found.append("humvee")
    return found


# The door echoes its own prompt into the figure line, and a full frame
# also sheds two-letter dust. Neither is stencil text.
ECHO_TOKENS = {"TEXT", "IN", "THE", "FIGURE", "OF", "AS"}


def keep_stencil(token: str) -> bool:
    """A plate token: not the prompt echo, and not a one- or two-letter speck."""
    if not token or token in ECHO_TOKENS:
        return False
    if len(token) < 2 or len(token) > 16:
        return False
    if any(ch.isdigit() for ch in token):
        return True
    return len(token) >= 4


def figure_words(text: str) -> list:
    """Stencil tokens from the door's 'Text in the figure' line."""
    match = re.search(r"Text in the figure:\s*(.*?)\s*$", text or "", re.I)
    if not match:
        return []
    words = []
    for raw in match.group(1).split():
        chars = [ch for ch in raw.upper() if ch.isalnum()]
        token = "".join(chars)
        if not keep_stencil(token) or token in words:
            continue
        words.append(token)
        if len(words) >= 8:
            break
    return words


class HaloFlorence:
    """One door call over a batch of stills. The model loads once per call."""

    def __init__(self, model_dir: str = ""):
        if not Path(DOOR_LIB).is_file():
            raise SystemExit(f"HaloCore door not found: {DOOR_LIB}")
        model = model_dir or os.environ.get("HALOCORE_FLORENCE_DIR") or DEFAULT_MODEL
        if not (Path(model) / "onnx" / "vision_encoder_int8.onnx").is_file():
            raise SystemExit(f"HaloCore Florence-2 not found: {model}")
        os.environ["HALOCORE_FLORENCE_DIR"] = model
        os.environ.setdefault("HALOCORE_FIGURES_QUANT", "int8")
        self.lib = ctypes.CDLL(DOOR_LIB)
        self.fn = self.lib.halocore_door_run
        self.fn.argtypes = [POINTER(_In), POINTER(_Out)]
        self.fn.restype = c_int32
        self.model = model

    def read_many(self, image_paths) -> dict:
        """Map each file stem to the door's caption text. Empty on failure."""
        paths = [str(p) for p in image_paths if p]
        if not paths:
            return {}
        print(f"[aerial] halocore florence {len(paths)} frame(s)", flush=True)
        parent = Path(tempfile.mkdtemp(prefix="eworld_florence_"))
        store = parent / "store"
        try:
            buf = (c_char_p * len(paths))(*[p.encode() for p in paths])
            req = _In(1, 0, buf, len(paths), str(store).encode(), 1)
            out = _Out()
            rc = self.fn(ctypes.byref(req), ctypes.byref(out))
            message = out.message.decode("utf-8", "replace").strip()
            if rc != 0 or not store.is_file():
                print(f"[aerial] halocore florence failed: {message or rc}", flush=True)
                return {}
            by_stem = {}
            for line in store.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stem = str(rec.get("doc") or "")
                text = str(rec.get("text") or "")
                if not stem or not text:
                    continue
                prev = by_stem.get(stem)
                by_stem[stem] = text if not prev else prev + "\n" + text
            return by_stem
        finally:
            shutil.rmtree(parent, ignore_errors=True)
            sibling = Path(str(store) + ".figures")
            if sibling.exists():
                shutil.rmtree(sibling, ignore_errors=True)
