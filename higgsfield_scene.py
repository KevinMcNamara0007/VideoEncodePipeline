#!/usr/bin/env python3
"""Submit the EWORLD range pass to Higgsfield Cinema Studio 4.0.

The still plate failed because the camera was told to hold and the vehicles
were told to creep. Each shot here sets pacing to dynamic, picks one moving
camera, and names what the assets do during the shot.

Auth, from https://console.higgsfield.ai :

    export HF_API_KEY_ID=...
    export HF_API_KEY_SECRET=...

Then:

    python3 higgsfield_scene.py            # submit, download, concat
    python3 higgsfield_scene.py --dry-run  # print the requests and exit

Shots land in higgsfield_shots/. The joined film lands in input/range_pass.mp4,
which ./run.sh will encode.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHOT_DIR = ROOT / "higgsfield_shots"
OUT = ROOT / "input" / "range_pass.mp4"
ENDPOINT = "https://api.higgsfield.ai/higgsfield/cinema-studio/4.0"

# Shared film setup. static-shot and slow-zoom are intentionally absent.
LOOK = {
    "genre": "action",
    "era": "2020s",
    "pacing": "dynamic",
    "light": "overhead-fall",
    "camera_model": "modern",
    "camera_lens": "clean-sharp",
    "camera_aperture": "f11-deep-focus",
    "color_palette": "field-post",
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "generate_audio": False,
}

# One action for the camera, one action for every asset. Markings are large
# and face the lens so a later 2 fps / 640 encode can still read them.
SHOTS = [
    {
        "name": "01_strip",
        "duration": 6,
        "camera_movement": "tracking",
        "prompt": (
            "Hyper-realistic aerial, bright daylight, arid training range. "
            "The drone flies forward fast and low over a dirt airstrip. "
            "The painted runway number 27 rushes under the lens and slides out "
            "the bottom of frame. A water tower marked RANGE 4 sweeps past on "
            "the right. Dust blows across the strip. Nothing is parked."
        ),
    },
    {
        "name": "02_column",
        "duration": 8,
        "camera_movement": "side-tracking",
        "prompt": (
            "Hyper-realistic side-tracking aerial. Five military trucks drive "
            "at road speed along a dirt road, wheels turning, dust trailing. "
            "The camera matches their speed so the stencil stays readable, "
            "then lets them pull ahead. Lead truck tailgate reads B127. "
            "Second vehicle bumper reads A14. Third truck door reads C330. "
            "Fourth bumper reads A22. The rear truck has no stencil. "
            "All five keep moving the whole shot."
        ),
    },
    {
        "name": "03_people",
        "duration": 6,
        "camera_movement": "truck-left",
        "prompt": (
            "Hyper-realistic aerial trucking left. The same truck column is "
            "still rolling. Three soldiers in uniform walk briskly beside the "
            "lead truck, full body, arms swinging, boots kicking dust. "
            "They do not stand still. The truck tailgate B127 stays visible "
            "as both the people and the trucks travel left to right."
        ),
    },
    {
        "name": "04_tracked",
        "duration": 6,
        "camera_movement": "truck-right",
        "prompt": (
            "Hyper-realistic aerial. A tracked armored vehicle drives quickly "
            "across the frame in profile, tracks visibly rotating, dust "
            "behind it. Turret side stencil T441 faces the camera and stays "
            "legible as it crosses. The truck column keeps driving in the "
            "background. The armored vehicle does not stop and does not fire."
        ),
    },
    {
        "name": "05_airplane",
        "duration": 6,
        "camera_movement": "pan-right",
        "prompt": (
            "Hyper-realistic aerial pan. A single-engine airplane crosses the "
            "upper frame left to right at speed, wings level, propeller "
            "spinning. Tail code AF218 is large and dark on a light tail. "
            "The ground and the truck column slide with the pan. The airplane "
            "exits frame before the shot ends."
        ),
    },
    {
        "name": "06_river",
        "duration": 8,
        "camera_movement": "helicopter-shot",
        "prompt": (
            "Hyper-realistic moving helicopter shot over a river at the edge "
            "of the range. A helicopter crosses the far background, rotor "
            "spinning. A rigid boat drives across the river leaving a wake, "
            "hull number P19 facing the camera. Both aircraft and boat are "
            "in motion the entire shot. The camera keeps translating."
        ),
    },
    {
        "name": "07_gate",
        "duration": 6,
        "camera_movement": "dolly-in",
        "prompt": (
            "Hyper-realistic aerial dolly in toward a concrete gate. A fuel "
            "truck drives through the gate while the camera closes. The sign "
            "above the gate reads NORTH GATE. Two crate stacks sit inside the "
            "wire, stenciled MK82 and 762, letters large. A plain concrete "
            "bridge is beyond the gate. The truck is moving as the shot ends."
        ),
    },
    {
        "name": "08_wide",
        "duration": 6,
        "camera_movement": "aerial-pullback",
        "prompt": (
            "Hyper-realistic aerial pullback, climbing fast. Below, the truck "
            "column is still driving, the tracked vehicle is still crossing, "
            "and the airplane is a small moving shape. The runway number 27 "
            "shrinks as the camera rises. Dust trails show the vehicles are "
            "moving. Cut while the climb is still underway."
        ),
    },
]


def credentials():
    key_id = os.environ.get("HF_API_KEY_ID", "").strip()
    secret = os.environ.get("HF_API_KEY_SECRET", "").strip()
    if key_id and secret:
        return key_id, secret
    packed = os.environ.get("HF_CREDENTIALS", "").strip()
    if ":" in packed:
        left, right = packed.split(":", 1)
        if left and right:
            return left, right
    sys.exit(
        "No Higgsfield credentials in the environment.\n"
        "Create a key at https://console.higgsfield.ai and export:\n"
        "  HF_API_KEY_ID\n"
        "  HF_API_KEY_SECRET\n"
        "Then run this script again. --dry-run prints the shots without a key."
    )


def request(method, url, body=None, auth=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if auth is not None:
        headers["Authorization"] = "Key %s:%s" % auth
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("Higgsfield %s %s -> %s\n%s" % (method, url, exc.code, detail))
    return json.loads(raw) if raw else {}


def submit_with_key(auth, shot):
    body = dict(LOOK)
    body["prompt"] = shot["prompt"]
    body["duration"] = shot["duration"]
    body["camera_movement"] = shot["camera_movement"]
    data = json.dumps(body).encode()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Key %s:%s" % auth,
        "Idempotency-Key": str(uuid.uuid4()),
    }
    req = urllib.request.Request(ENDPOINT, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(
            "Higgsfield rejected %s -> %s\n%s" % (shot["name"], exc.code, detail)
        )
    return payload, body


def poll(auth, status_url, name):
    delay = 3.0
    while True:
        status = request("GET", status_url, auth=auth)
        state = status.get("status", "")
        print("[higgsfield] %s %s" % (name, state), flush=True)
        if state == "completed":
            video = status.get("video") or {}
            url = video.get("url")
            if not url:
                raise SystemExit("completed with no video url:\n%s" % json.dumps(status)[:800])
            return url
        if state in ("failed", "nsfw", "canceled", "cancelled", "error"):
            raise SystemExit("%s ended as %s\n%s" % (name, state, json.dumps(status)[:800]))
        time.sleep(delay)
        delay = min(delay * 1.4, 15.0)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def concat(paths, dest):
    listing = dest.with_suffix(".txt")
    with listing.open("w") as fh:
        for path in paths:
            fh.write("file '%s'\n" % path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(dest),
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        # Mixed timebases from separate generations often refuse -c copy.
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-pix_fmt", "yuv420p", str(dest),
            ],
            capture_output=True,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit("ffmpeg concat failed")
    listing.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audio", action="store_true", help="ask Cinema Studio for a soundtrack")
    args = ap.parse_args()
    if args.audio:
        LOOK["generate_audio"] = True

    if args.dry_run:
        for shot in SHOTS:
            body = dict(LOOK)
            body.update({
                "prompt": shot["prompt"],
                "duration": shot["duration"],
                "camera_movement": shot["camera_movement"],
            })
            print("--- %s ---" % shot["name"])
            print(json.dumps(body, indent=2))
        print("endpoint", ENDPOINT)
        return

    auth = credentials()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for shot in SHOTS:
        dest = SHOT_DIR / ("%s.mp4" % shot["name"])
        if dest.is_file() and dest.stat().st_size > 0:
            print("[higgsfield] reuse %s" % dest.name, flush=True)
            paths.append(dest)
            continue
        payload, _body = submit_with_key(auth, shot)
        status_url = payload.get("status_url")
        if not status_url:
            raise SystemExit("no status_url:\n%s" % json.dumps(payload)[:800])
        url = poll(auth, status_url, shot["name"])
        download(url, dest)
        print("[higgsfield] saved %s" % dest, flush=True)
        paths.append(dest)
    concat(paths, OUT)
    print("[higgsfield] film %s" % OUT)


if __name__ == "__main__":
    main()
