# EWORLD test scene

The plate has to move. A locked camera and parked vehicles produce an empty encode. `higgsfield_scene.py` submits these beats to Cinema Studio 4.0 with `pacing: dynamic` and a camera move on every shot. Vehicles drive, people walk, aircraft cross.

Aerial pass over a daylight training range. Markings are large and face the lens long enough for a 2 fps, 640×640 sample to read them.

`yolov8n` tokens this scene can produce: `truck`, `car`, `person`, `airplane`, `boat`, plus `ocr:` strings and one `scene:` tag per frame. `tank`, `humvee`, `apc`, and `artillery` appear only when the checkpoint already has those class names. Until then the tracked vehicle is still in the shot so a later model can be scored on the same plate.

## 0:00–0:20 — Landmark

The drone is high, nose down about 30 degrees, moving forward slowly. Below is a dirt strip. The near end is painted with a white runway number, `27`, two meters tall. To the right of the strip a water tower carries `RANGE 4` in white block letters on a dark tank.

Nothing else is in frame. The hold is the whole 20 seconds.

Expect: `ocr:27`, `ocr:RANGE4`. No vehicle class.

## 0:20–0:50 — Column

Descend and level off behind a column on the dirt road that leaves the strip. Five vehicles, evenly spaced, driving away from the camera at a walk.

1. Lead cargo truck. Tailgate stencil `B127`, white on dark, letters taller than a hand.
2. Humvee-type soft truck. Bumper number `A14`.
3. Second cargo truck. Door stencil `C330`.
4. Humvee-type soft truck. Bumper number `A22`.
5. Rear cargo truck. No readable stencil. It is there so a frame can contain several trucks at once.

The road fills the lower half of the frame. The stencils face the camera as the drone closes.

Expect: `truck` and `car`, `scene:ground`, `ocr:B127`, `ocr:A14`, `ocr:C330`, `ocr:A22`.

## 0:50–1:10 — People with the vehicles

The column stops. Two people in uniform step out of the lead truck and stand beside the rear wheels, full body, not obscured, not in contact. A third person stays in the bed of truck 3, visible from the waist up. They wait. The drone holds.

Expect: `person` with `truck`, `scene:ground-with-person`. The same stencil tokens as the previous beat if the tailgate is still in frame.

## 1:10–1:30 — Tracked vehicle

A tracked armored vehicle crosses left to right on a side road, in profile, filling about a third of the frame width. Turret side carries `T441` in light paint. It does not fire. The column is still in the background so the frame contains both the tracked vehicle and at least one truck.

Expect with `yolov8n`: `truck` on the column. The tracked vehicle has no COCO class. Expect with a checkpoint that knows it: `tank` or `apc`, and `scene:armor`. Either way: `ocr:T441`.

## 1:30–1:50 — Fixed wing

The drone pitches up. A single-engine airplane crosses the upper third of the frame, left to right, wings level, close enough that the wingspan is about a quarter of the frame. The tail is light colored with the dark code `AF218`. The ground column is small at the bottom edge or just out of frame.

Expect: `airplane`, `scene:air`, `ocr:AF218`.

## 1:50–2:10 — Second aircraft and a boat

The drone yaws toward a river at the edge of the range. A helicopter crosses farther away, smaller than the airplane, rotor visible. On the river a rigid boat sits side-on to the camera. The hull, facing the lens, reads `P19`. The helicopter has no stencil.

Expect: `airplane` if the helicopter is classified as an airplane, `boat`, `ocr:P19`. `scene:water` when the boat is the only vehicle. `scene:air` when the aircraft is alone in frame.

## 2:10–2:30 — Gate and stores

Descend to the north fence. A concrete gate house. The sign above the gate reads `NORTH GATE`. Inside the wire, two stacks of crates. The near stack is stenciled `MK82`. The far stack is stenciled `762`. A fuel truck is parked at the gate, no people. The bridge just beyond the gate is plain concrete with no text.

Expect: `truck` on the fuel truck, `ocr:NORTHGATE`, `ocr:MK82`, `ocr:762`. The bridge is in frame as infrastructure and has no class token.

## 2:30–2:40 — Wide

Climb back to the opening height. The strip with `27`, the column, the tracked vehicle, and the gate are all visible, small. Hold ten seconds and cut.

Expect: whatever classes are still large enough to detect, and `ocr:27` if the runway number is still legible. This beat is the wide context, not the OCR beat.

## How to shoot it so the encode can read it

- Stencil letters stay at least 32 pixels tall after the frame is scaled to 640×640. If a marking is a speck, Tesseract will not emit a token.
- Each marking is held, facing the camera, for at least one second.
- Light letters on a dark surface, or dark letters on a light surface. The reader accepts either.
- Sample rate for this test is 2 fps. Fast crosses will be missed. The airplane and the tracked vehicle should take about five seconds to cross the frame.
- Drop the file in `input/` as an `.mp4`. `./run.sh` encodes it into `out/`.
