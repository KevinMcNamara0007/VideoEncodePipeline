#!/usr/bin/env python3
"""Thirty Seedance 2.5 shots for the EWORLD aerial.

Each clip is one cut: the camera moves, and the vehicles, people, or aircraft
move with it. Together they cover trucks, humvees, tanks, people, weapons on
the move, marked munitions, infrastructure, landmarks, and air and river craft.
Readable stencils are large and face the lens.

Uses the client the way the Seedance text-to-video example does:

    from dotenv import load_dotenv
    import higgsfield_client
    load_dotenv(".env.local")
    higgsfield_client.subscribe("bytedance/seedance-2.5/text-to-video", arguments={...})

Credentials in .env.local, as HF_KEY or as HF_API_KEY plus HF_API_SECRET.
HF_API_KEY_ID and HF_API_KEY_SECRET are accepted too.

    python3 higgsfield_30.py --dry-run
    python3 higgsfield_30.py --limit 1
    python3 higgsfield_30.py
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
import higgsfield_client

ROOT = Path(__file__).resolve().parent
SHOT_DIR = ROOT / "higgsfield_shots" / "seedance"
FILM = ROOT / "input" / "range_pass.mp4"
MODEL = "bytedance/seedance-2.5/text-to-video"

LOOK = (
    "Photoreal Hollywood aerial, large-format cinema camera, hard desert daylight, "
    "real dust, real weight, natural motion blur, sharp focus. "
    "Stencil letters are tall, high-contrast, and face the lens. "
    "The camera is moving and the subject is moving when the shot cuts. "
    "No slow motion, no freeze frame, no locked-off camera, no parked vehicles, "
    "no people standing still."
)

# name, duration, the action. Duration is 5 to 8 seconds. Seedance allows 4 to 30.
CLIPS = [
    ("01_runway", 5,
     "A drone races forward low over a dirt airstrip. The painted runway number 27 "
     "rushes under the lens and slides out the bottom of the frame. Dust blows across the strip."),
    ("02_tower", 5,
     "The drone tracks sideways at speed past a water tower. White block letters RANGE 4 "
     "slide across the frame and stay sharp. The tower drops behind as the drone keeps flying."),
    ("03_ridge_sign", 5,
     "A cargo truck drives fast along a ridge road toward a sign that reads NORTH RIDGE. "
     "The drone flies alongside. The sign grows, passes, and the truck continues."),
    ("04_truck_b127", 6,
     "A military cargo truck drives toward the camera at road speed, wheels turning, dust trailing. "
     "The tailgate stencil B127 stays readable as the truck fills the frame and passes under the drone."),
    ("05_truck_c330", 5,
     "A second cargo truck overtakes on the left. Door stencil C330 faces the camera. "
     "Both trucks keep driving. The drone pans to stay with the overtaker."),
    ("06_truck_convoy", 8,
     "Five cargo trucks drive in column at road speed on a dirt road. "
     "The drone side-tracks the column. Dust links the trucks. None of them stop."),
    ("07_humvee_a14", 6,
     "A humvee drives across the frame, wheels turning, bumper number A14 large and white. "
     "The drone trucks right to match it, then the humvee pulls ahead."),
    ("08_humvee_a22", 5,
     "A humvee with bumper number A22 turns off the road onto open dirt without stopping. "
     "The drone descends and follows through the turn. Dust blooms off the rear tires."),
    ("09_humvee_column", 6,
     "Three humvees drive in file toward the horizon. The drone flies up the column from behind. "
     "Each vehicle is moving. The nearest bumper reads A14."),
    ("10_tank_profile", 6,
     "A main battle tank drives in profile at speed, tracks visibly rotating, dust behind the skirts. "
     "Turret stencil T441 faces the camera and stays legible as the tank crosses."),
    ("11_tank_crosses_trucks", 6,
     "The tank drives left to right across the front of the moving truck column. "
     "Tracks turn. Trucks keep rolling behind it. Turret side still reads T441."),
    ("12_tank_headon", 5,
     "The tank drives toward the low drone, growing in frame, tracks throwing dust. "
     "The drone climbs as the tank passes underneath. The hull number T441 flashes on the side during the pass."),
    ("13_soldiers_run", 5,
     "Three soldiers in uniform run at a full stride beside the lead truck, arms swinging, boots in the dust. "
     "The truck drives with them. The drone tracks the group. Nobody stands still."),
    ("14_soldiers_board", 6,
     "Two soldiers jog alongside a rolling cargo truck and pull themselves up onto the side step while it is still moving. "
     "The truck does not stop. The drone flies parallel."),
    ("15_soldier_in_bed", 5,
     "A soldier rides in the open bed of a driving cargo truck, bracing as the truck hits ruts. "
     "The truck tailgate B127 is visible. The drone stays off the rear quarter."),
    ("16_turret_traverse", 5,
     "While the tank drives, its turret rotates in a smooth traverse. Tracks keep turning. "
     "The drone orbits just enough to show both the hull moving and the turret turning. Stencil T441 remains readable."),
    ("17_mounted_gun", 5,
     "A humvee drives at speed with a roof-mounted machine gun. The gun swings on its mount as the vehicle turns. "
     "The drone tracks the humvee. Bumper A22 stays in frame. The gun is not firing."),
    ("18_rifles_carried", 5,
     "Four soldiers run across open ground carrying rifles slung across their chests, kits bouncing. "
     "A humvee drives past them in the same direction. The drone flies low and fast alongside the runners."),
    ("19_crates_mk82", 6,
     "A cargo truck rolls slowly, then pulls away, while two soldiers slide wooden crates up the tailgate ramp. "
     "The near crate is stenciled MK82 in tall white letters. Crates, soldiers, and the truck are all in motion."),
    ("20_pallet_762", 5,
     "A trailer stacked with crates stenciled 762 is towed along the dirt road by a driving truck. "
     "The drone drops beside the trailer so the stencil faces the lens, then the trailer moves ahead."),
    ("21_ammo_truck", 5,
     "An ammunition truck with AMMO on the cargo door drives through frame at road speed. "
     "The drone pans with it. Dust, turning wheels, the lettering sharp as it passes."),
    ("22_bridge", 6,
     "The truck column drives onto and across a concrete bridge without slowing. "
     "The drone flies beside the span. Vehicles enter one end and exit the other during the shot."),
    ("23_north_gate", 5,
     "A fuel truck drives through a concrete gate. The sign above the gate reads NORTH GATE in large letters. "
     "The drone dollies in as the truck passes under the sign and continues inside the wire."),
    ("24_fuel_point", 5,
     "A fuel truck drives up to rubber fuel bladders marked FUEL and rolls past them without parking. "
     "The drone tracks the truck. The FUEL lettering slides through frame."),
    ("25_airplane", 5,
     "A single-engine airplane crosses the upper frame left to right, propeller spinning. "
     "Tail code AF218 is large on a light tail. The drone pans hard to follow, and the airplane exits frame."),
    ("26_helicopter", 6,
     "A helicopter flies low across the range, rotor blurred with motion, nose slightly down. "
     "The drone yaws to track it. The helicopter crosses the whole frame and is still flying at the cut."),
    ("27_second_aircraft", 5,
     "Higher and farther, a second fixed-wing aircraft crosses the opposite direction. "
     "The drone tilts up and pans to follow. The first helicopter is a small moving shape below."),
    ("28_boat", 6,
     "A rigid boat drives across a river at the edge of the range, throwing a wake. "
     "Hull number P19 faces the camera. The drone flies along the river matching the boat."),
    ("29_mixed_pass", 8,
     "The drone races forward along the dirt road. Trucks and humvees drive below. "
     "The tank crosses a side road. The helicopter moves across the top of frame. Dust trails show every vehicle is moving."),
    ("30_pullback", 6,
     "The drone climbs fast in a pullback. Below, the column is still driving, the tank is still crossing, "
     "and the airplane is a small moving shape. Runway 27 shrinks. The climb is still underway at the cut."),
]


def load_credentials():
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
    if not os.getenv("HF_API_KEY") and os.getenv("HF_API_KEY_ID"):
        os.environ["HF_API_KEY"] = os.environ["HF_API_KEY_ID"]
    if not os.getenv("HF_API_SECRET") and os.getenv("HF_API_KEY_SECRET"):
        os.environ["HF_API_SECRET"] = os.environ["HF_API_KEY_SECRET"]
    packed = os.getenv("HF_CREDENTIALS", "")
    if packed and ":" in packed and not os.getenv("HF_KEY") and not os.getenv("HF_API_KEY"):
        key_id, secret = packed.split(":", 1)
        os.environ["HF_API_KEY"] = key_id
        os.environ["HF_API_SECRET"] = secret


def arguments(prompt, duration):
    return {
        "prompt": prompt + " " + LOOK,
        "duration": duration,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "bitrate_mode": "high",
        "output_format": "mp4",
        "generate_audio": True,
    }


def video_url(result):
    if isinstance(result, dict):
        video = result.get("video")
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
        if isinstance(video, str):
            return video
        for key in ("url", "video_url"):
            if result.get(key):
                return result[key]
    raise SystemExit("no video url in result: %s" % str(result)[:500])


def concat(paths, dest):
    listing = dest.with_suffix(".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with listing.open("w") as fh:
        for path in paths:
            fh.write("file '%s'\n" % path)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit("ffmpeg concat failed")
    listing.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="generate only the first N clips")
    args = ap.parse_args()

    clips = CLIPS if args.limit <= 0 else CLIPS[: args.limit]
    if len(CLIPS) != 30:
        raise SystemExit("expected 30 clips, found %d" % len(CLIPS))

    if args.dry_run:
        for name, duration, action in clips:
            body = arguments(action, duration)
            print("--- %s  %ss ---" % (name, duration))
            print(body["prompt"])
            print()
        print("%d clips, model %s" % (len(clips), MODEL))
        return

    load_credentials()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, duration, action) in enumerate(clips, start=1):
        dest = SHOT_DIR / ("%s.mp4" % name)
        if dest.is_file() and dest.stat().st_size > 0:
            print("[seedance] reuse %s" % dest.name, flush=True)
            paths.append(dest)
            continue
        print("[seedance] %d/%d %s" % (index, len(clips), name), flush=True)
        result = higgsfield_client.subscribe(
            MODEL,
            arguments=arguments(action, duration),
            on_queue_update=lambda status, name=name: print(
                "[seedance] %s %s" % (name, getattr(status, "status", status)),
                flush=True,
            ),
        )
        url = video_url(result)
        urllib.request.urlretrieve(url, dest)
        print("[seedance] saved %s" % dest, flush=True)
        paths.append(dest)

    if len(paths) == len(CLIPS):
        concat(paths, FILM)
        print("[seedance] film %s" % FILM)
    else:
        print("[seedance] %d of 30 shots on disk. Re-run with no --limit to join the film." % len(paths))


if __name__ == "__main__":
    main()
