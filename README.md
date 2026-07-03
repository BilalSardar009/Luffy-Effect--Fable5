# Luffy Effect — rubber-stretch webcam filter

The classic *One Piece* Luffy cheek-stretch, live on your webcam, built with
**MediaPipe + Python + OpenCV**.

Pinch your cheek on camera (thumb + index finger together) and pull:
the skin stretches like rubber and follows your fingers, ending exactly
where your hand stops. Let go and it snaps back with an elastic wobble.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python luffy_effect.py
```

The first run downloads two small MediaPipe model files
(`face_landmarker.task`, `hand_landmarker.task`) into `models/` — after
that it works offline.

## How to use it

1. Face the camera so your face is detected.
2. Bring your **open** hand to your cheek, then pinch thumb + index
   together right on the spot you want to grab. (The open-hand-first rule
   is what stops accidental grabs while your hand is moving.)
3. Keep them pinched and pull away — your cheek stretches with your hand
   and stays wherever your hand stops. Nothing warps until you actually
   pull; just pinching leaves your face alone.
4. Open your fingers to release — the skin snaps **straight back,
   overshoots past where it started, and swings to rest**, like a rubber
   band.
5. Pinch near your **other hand** instead and you pull a **thin elastic
   strand** off your finger — same elastic snap on release.

Works with **both hands at once** (stretch both cheeks!), and you can also
grab and stretch anything else in the frame, not just your face.

### Keys

| Key | Action |
| --- | ------ |
| `q` / `ESC` | quit |
| `d` | debug overlay (landmarks, grab anchor → pull point, influence ellipse) |
| `o` | toggle drawing your real hand on top of the warp |
| `m` | toggle mirror (selfie) view |

### Options

```
--camera N            camera index (default 0; try 1 if the window is black)
--width / --height    capture size (default 960x540; lower it if FPS is low)
--max-stretch F       max stretch length in face-widths (default 3.0)
--model-complexity M  hand model 0=faster 1=more accurate (legacy API only)
--selftest [OUT.png]  run the warp math self-test without a camera and exit
```

## How it works

- **Face**: MediaPipe FaceLandmarker gives 478 face landmarks per frame.
- **Hands**: MediaPipe HandLandmarker gives 21 landmarks for up to 2 hands.
- **Pinch detection**: thumb-tip to index-tip distance, normalized by hand
  size, with hysteresis, a hold-to-engage debounce, and an
  open-hand-before-grab rule so a hand approaching the face never grabs by
  accident.
- **Grabbing**: the grab starts with *zero* displacement exactly at the
  pinch point, and only stretches once the hand moves past a small dead
  zone. Near the face the anchor is stored relative to the nearest face
  landmark, so the grab stays glued to your cheek even while your head
  moves. A pinch away from the face grabs that spot in the frame instead.
- **The stretch** is rendered in two layers, like the original viral video:
  - A **skin flap overlay**: a strip of *real* skin sampled around the grab
    point, stretched uniformly along the pull, drawn on top of the frame
    with feathered edges, a rounded tip, a specular sheen, and slight
    brightening — a clean rubber flap, with zero background smearing
    because the frame under it is never liquified.
  - A **subtle root warp** (`cv2.remap` with a small cone-shaped falloff)
    that tugs the cheek toward the pull so the flap looks attached.
- **Snap-back**: on release, a damped spring animates the stretch back to
  zero with an overshoot wobble — the rubber-band feel.
- **Hand on top**: after warping, the real (unwarped) hand pixels are
  composited back over the frame with a feathered mask, so your fingers
  appear to be holding the stretched skin.

## Troubleshooting

- **`Could not open camera 0`** — try `--camera 1` (or 2), and check OS
  camera permissions for your terminal.
- **`libGLESv2.so.2: cannot open shared object file`** (headless Linux) —
  `sudo apt-get install libgles2 libegl1 libgl1`.
- **Low FPS** — run with `--width 640 --height 360`.
- **No camera handy?** — `python luffy_effect.py --selftest out.png` checks
  the warp math and renders a before/during/after montage.
