#!/usr/bin/env python3
"""Microsoft Florence-2 for labels YOLOv8 did not emit, plus region OCR.

The vendored int8 ONNX vision encoder uses grouped ConvInteger, which the
local ONNX Runtime build does not implement. This loads the PyTorch
Florence-2-base checkpoint instead (same Microsoft model, same tasks).
"""

import os
import sys
from pathlib import Path

# Set before the first torch import. Florence-2 generate calls aten::isin,
# which MPS does not implement. The fallback runs that one op on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Phrases the COCO head cannot name. YOLO-World is asked for the same
# military set; Florence only keeps a hit when that label is still absent.
GAP_PHRASES = (
    "tank",
    "humvee",
    "helicopter",
    "rifle",
    "machine gun",
    "artillery",
    "crate",
    "bridge",
    "boat",
    "water tower",
)

# Canonical phrase -> index token.
PHRASE_TOKEN = {
    "tank": "tank",
    "humvee": "humvee",
    "helicopter": "helicopter",
    "rifle": "weapon:rifle",
    "machine gun": "weapon:machine-gun",
    "artillery": "artillery",
    "crate": "crate",
    "bridge": "bridge",
    "boat": "boat",
    "water tower": "water-tower",
}

FLORENCE_CONF = 0.5
MAX_GAP_BOXES = 8


def _canon(label: str) -> str:
    words = (label or "").lower().strip().split()
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


def quad_to_xyxy(quad):
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


class FlorenceGap:
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        path = str(Path(model_dir))
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self._gap_prompt = "<OPEN_VOCABULARY_DETECTION>" + "<and>".join(GAP_PHRASES)

    def _run(self, pil, prompt: str, task: str, max_new_tokens: int):
        try:
            return self._generate(pil, prompt, task, max_new_tokens)
        except Exception as exc:
            if self.device != "mps" or "isin" not in str(exc).lower():
                raise
            sys.stderr.write(
                "[florence] MPS rejected aten::isin; Florence-2 is now on CPU\n"
            )
            self.model = self.model.to("cpu")
            self.device = "cpu"
            return self._generate(pil, prompt, task, max_new_tokens)

    def _generate(self, pil, prompt: str, task: str, max_new_tokens: int):
        torch = self.torch
        inputs = self.processor(text=prompt, images=pil, return_tensors="pt")
        moved = {}
        for key, val in inputs.items():
            moved[key] = val.to(self.device) if hasattr(val, "to") else val
        with torch.inference_mode():
            ids = self.model.generate(
                input_ids=moved["input_ids"],
                pixel_values=moved["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=task, image_size=(pil.width, pil.height)
        )
        return parsed.get(task) or {}

    def read(self, bgr, already: set):
        """Boxes Florence adds. `already` holds class tokens YOLOv8 already kept."""
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes = []
        try:
            gaps = self._run(
                pil, self._gap_prompt, "<OPEN_VOCABULARY_DETECTION>", 64
            )
        except Exception as exc:
            gaps = {}
            boxes.append({"_error": f"gap: {exc}"})
        labels = gaps.get("bboxes_labels") or []
        bboxes = gaps.get("bboxes") or []
        for label, xy in zip(labels, bboxes):
            token = PHRASE_TOKEN.get(_canon(label))
            if not token or token in already:
                continue
            if any(b.get("class") == token for b in boxes):
                continue
            if len(xy) != 4:
                continue
            boxes.append({
                "class": token,
                "confidence": FLORENCE_CONF,
                "xyxy": [float(v) for v in xy],
                "ocr": [],
                "source": "florence",
            })
            if sum(1 for b in boxes if b.get("source") == "florence" and not str(b.get("class", "")).startswith("ocr:")) >= MAX_GAP_BOXES:
                break

        try:
            ocr = self._run(pil, "<OCR_WITH_REGION>", "<OCR_WITH_REGION>", 128)
        except Exception as exc:
            ocr = {}
            boxes.append({"_error": f"ocr: {exc}"})
        words = ocr.get("labels") or []
        quads = ocr.get("quad_boxes") or []
        seen = set()
        for word, quad in zip(words, quads):
            chars = [ch for ch in str(word).upper() if ch.isalnum()]
            if not 2 <= len(chars) <= 16:
                continue
            token = "ocr:" + "".join(chars)
            if token in seen or token in already:
                continue
            seen.add(token)
            try:
                xy = quad_to_xyxy(quad)
            except (TypeError, ValueError, IndexError):
                continue
            boxes.append({
                "class": token,
                "confidence": FLORENCE_CONF,
                "xyxy": xy,
                "ocr": [token[4:]],
                "source": "florence-ocr",
            })
            if len(seen) >= 8:
                break
        return boxes
