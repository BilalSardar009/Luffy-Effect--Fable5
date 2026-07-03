#!/usr/bin/env python3
"""Luffy rubber-stretch webcam effect (MediaPipe + OpenCV).

Pinch your cheek (thumb + index finger together) on camera and pull:
the skin stretches like rubber and follows your fingers, ending exactly
where your hand stops. Release the pinch and it snaps back with an
elastic wobble -- just like Luffy.

Run:
    python luffy_effect.py

Keys:
    q / ESC  quit
    d        toggle debug overlay (landmarks, grab lines)
    o        toggle drawing your real hand on top of the warp
    m        toggle mirror (selfie) view
"""

import argparse
import math
import os
import time
import urllib.request

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Warp core (pure OpenCV/numpy -- no MediaPipe needed here)
# ----------------------------------------------------------------------------

# Displacement maps are computed on a downscaled grid and resized back up,
# which keeps the per-frame warp cheap enough for real time.
MAP_SCALE = 0.25


def make_grab(anchor, pull, base_radius):
    """Build one grab tuple: pull the pixel at `anchor` to `pull`.

    The influence region is an ellipse around the pull point, elongated
    along the pull direction, so a long stretch forms a narrow rubber flap
    (like a pinched cheek) instead of smearing a huge circle of the frame.
    """
    stretch = math.hypot(pull[0] - anchor[0], pull[1] - anchor[1])
    r_behind = max(30.0, base_radius * 0.8 + 0.75 * stretch)  # toward anchor
    r_ahead = max(30.0, base_radius * 0.8)   # past the fingertip: stay short
    r_perp = max(30.0, base_radius * 0.8 + 0.15 * stretch)
    return (anchor[0], anchor[1], pull[0], pull[1], r_behind, r_ahead, r_perp)


def build_maps(h, w, grabs, scale=MAP_SCALE):
    """Build cv2.remap coordinate maps for a set of rubber grabs.

    Each grab is (px, py, qx, qy, r_behind, r_ahead, r_perp): the point
    (px, py) is pulled to (qx, qy). Inverse mapping with a smooth compact
    falloff so the pixel that lands exactly at the fingertip is the one that
    was grabbed. The falloff reaches r_behind back toward the anchor (the
    rubber flap) but only r_ahead past the fingertip, so the background in
    front of the hand barely smears.
    """
    sw = max(2, int(w * scale))
    sh = max(2, int(h * scale))
    # Sample positions must match cv2.resize's pixel-center convention, or
    # upscaling the maps shifts the whole frame instead of just the grabs.
    xs = (np.arange(sw, dtype=np.float32) + 0.5) * (w / sw) - 0.5
    ys = (np.arange(sh, dtype=np.float32) + 0.5) * (h / sh) - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Only the displacement delta is computed at low res: it is zero away
    # from the grabs, so upscaling it cannot disturb the rest of the frame.
    delta_x = np.zeros((sh, sw), np.float32)
    delta_y = np.zeros((sh, sw), np.float32)

    for px, py, qx, qy, r_behind, r_ahead, r_perp in grabs:
        vx, vy = qx - px, qy - py
        length = math.hypot(vx, vy)
        ux, uy = (vx / length, vy / length) if length > 1e-3 else (1.0, 0.0)
        dx = grid_x - qx
        dy = grid_y - qy
        along = dx * ux + dy * uy      # distance along the pull direction
        perp = dy * ux - dx * uy       # distance across it
        # (x/r)^2 has zero slope at x=0, so switching radii there is seamless.
        r_along = np.where(along > 0, np.float32(r_ahead), np.float32(r_behind))
        k = 1.0 - (along / r_along) ** 2 - (perp / r_perp) ** 2
        np.clip(k, 0.0, None, out=k)
        k *= k  # smooth falloff, exactly 1 at the pull point, 0 outside
        delta_x -= k * vx
        delta_y -= k * vy

    if (sw, sh) != (w, h):
        delta_x = cv2.resize(delta_x, (w, h), interpolation=cv2.INTER_LINEAR)
        delta_y = cv2.resize(delta_y, (w, h), interpolation=cv2.INTER_LINEAR)
    full_x, full_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    return full_x + delta_x, full_y + delta_y


def apply_warp(frame, grabs):
    if not grabs:
        return frame
    h, w = frame.shape[:2]
    map_x, map_y = build_maps(h, w, grabs)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


class SnapBack:
    """Damped spring that wobbles a released grab back to its anchor."""

    DECAY = 8.0        # 1/s exponential decay
    FREQ_HZ = 3.2      # wobble frequency
    DONE_BELOW = 0.02  # amplitude at which the wobble is considered finished

    def __init__(self, anchor, release_point, base_radius):
        self.anchor = anchor
        self.v0 = (release_point[0] - anchor[0], release_point[1] - anchor[1])
        self.base_radius = base_radius
        self.t0 = time.time()

    def grab(self):
        """Current grab tuple for this wobble, or None once it has settled."""
        t = time.time() - self.t0
        s = math.exp(-t * self.DECAY) * math.cos(2 * math.pi * self.FREQ_HZ * t)
        if abs(s) < self.DONE_BELOW:
            return None
        px, py = self.anchor
        return make_grab((px, py),
                         (px + self.v0[0] * s, py + self.v0[1] * s),
                         self.base_radius)


# ----------------------------------------------------------------------------
# Face/hand tracking (MediaPipe imported lazily so --selftest works without it)
# ----------------------------------------------------------------------------

PINCH_ENGAGE = 0.40   # pinch when tip distance / hand size drops below this
PINCH_RELEASE = 0.55  # release when it rises above this (hysteresis)
THUMB_TIP, INDEX_TIP = 4, 8
WRIST, MIDDLE_MCP = 0, 9
# Face mesh side points used to estimate face width in pixels.
FACE_LEFT, FACE_RIGHT = 234, 454

_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
FACE_MODEL_URL = (_MODEL_BASE +
                  "/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
HAND_MODEL_URL = (_MODEL_BASE +
                  "/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


def _ensure_model(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)} (one-time) ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


class TaskTrackers:
    """Face + hand tracking via the modern MediaPipe Tasks API (>=0.10)."""

    def __init__(self, num_hands=2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models")
        face_model = _ensure_model(FACE_MODEL_URL,
                                   os.path.join(models_dir, "face_landmarker.task"))
        hand_model = _ensure_model(HAND_MODEL_URL,
                                   os.path.join(models_dir, "hand_landmarker.task"))

        self._mp = mp
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_model),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            ))
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
            ))
        self._last_ts = -1

    def process(self, rgb):
        """Return (face_pts, hands): normalized face landmarks (Nx2 or None)
        and a list of (21x2 normalized landmarks, handedness label)."""
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:  # VIDEO mode needs strictly increasing stamps
            ts = self._last_ts + 1
        self._last_ts = ts

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        face_res = self._face.detect_for_video(image, ts)
        hand_res = self._hands.detect_for_video(image, ts)

        face_pts = None
        if face_res.face_landmarks:
            face_pts = np.float32([(lm.x, lm.y)
                                   for lm in face_res.face_landmarks[0]])
        hands = []
        for i, lms in enumerate(hand_res.hand_landmarks):
            label = str(i)
            if hand_res.handedness and len(hand_res.handedness) > i:
                label = hand_res.handedness[i][0].category_name
            hands.append((np.float32([(lm.x, lm.y) for lm in lms]), label))
        return face_pts, hands

    def close(self):
        self._face.close()
        self._hands.close()


class SolutionsTrackers:
    """Fallback for older MediaPipe versions that still ship mp.solutions."""

    def __init__(self, num_hands=2, model_complexity=1):
        import mediapipe as mp
        self._face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=num_hands, model_complexity=model_complexity,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, rgb):
        rgb.flags.writeable = False
        face_res = self._face.process(rgb)
        hand_res = self._hands.process(rgb)
        rgb.flags.writeable = True

        face_pts = None
        if face_res.multi_face_landmarks:
            face_pts = np.float32([
                (lm.x, lm.y)
                for lm in face_res.multi_face_landmarks[0].landmark])
        hands = []
        if hand_res.multi_hand_landmarks:
            for i, lm in enumerate(hand_res.multi_hand_landmarks):
                label = str(i)
                if hand_res.multi_handedness:
                    label = hand_res.multi_handedness[i].classification[0].label
                hands.append((np.float32([(p.x, p.y) for p in lm.landmark]),
                              label))
        return face_pts, hands

    def close(self):
        self._face.close()
        self._hands.close()


def make_trackers(args):
    import mediapipe as mp
    if hasattr(mp, "tasks"):
        return TaskTrackers(num_hands=2)
    if hasattr(mp, "solutions"):
        return SolutionsTrackers(num_hands=2,
                                 model_complexity=args.model_complexity)
    raise SystemExit("Unsupported mediapipe build: neither Tasks API nor "
                     "legacy solutions found. Try: pip install -U mediapipe")


class HandGrab:
    """Pinch-grab state for one tracked hand."""

    def __init__(self):
        self.active = False
        self.face_idx = None    # face landmark being held, if snapped to face
        self.fixed_anchor = None  # anchor for grabs that start off the face
        self.q = None           # smoothed pull point (fingertips)

    def anchor(self, face_pts):
        if self.face_idx is not None and face_pts is not None:
            return tuple(face_pts[self.face_idx])
        return self.fixed_anchor

    def engage(self, pinch_pt, face_pts, snap_dist):
        self.active = True
        self.q = pinch_pt
        self.face_idx = None
        self.fixed_anchor = pinch_pt
        if face_pts is not None:
            d = np.linalg.norm(face_pts - np.float32(pinch_pt), axis=1)
            idx = int(np.argmin(d))
            if d[idx] <= snap_dist:
                self.face_idx = idx
                self.fixed_anchor = None

    def update(self, pinch_pt, smoothing=0.55):
        qx = self.q[0] * (1 - smoothing) + pinch_pt[0] * smoothing
        qy = self.q[1] * (1 - smoothing) + pinch_pt[1] * smoothing
        self.q = (qx, qy)

    def release(self):
        self.active = False
        self.face_idx = None
        self.fixed_anchor = None
        self.q = None


def hand_pinch_info(pts):
    """Return (pinch_point_px, pinch_ratio) for one hand's pixel landmarks."""
    hand_size = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if hand_size < 1e-3:
        return None
    tip_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
    pinch_pt = tuple((pts[THUMB_TIP] + pts[INDEX_TIP]) / 2.0)
    return pinch_pt, tip_dist / hand_size


def hand_overlay_mask(hand_pts_list, h, w):
    """Feathered mask covering the detected hands, so the real (unwarped)
    fingers are drawn on top of the stretched skin."""
    mask = np.zeros((h, w), np.uint8)
    for pts in hand_pts_list:
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    if not mask.any():
        return None
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask.astype(np.float32)[..., None] / 255.0


def run_live(args):
    trackers = make_trackers(args)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. "
            "Try --camera 1 (or another index) and check camera permissions."
        )

    mirror = True
    debug = False
    hand_overlay = True
    grabs_by_hand = {}   # handedness label -> HandGrab
    snapbacks = []
    fps, t_prev = 0.0, time.time()

    print("Luffy effect running. Pinch your cheek and pull! (q to quit)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_pts, tracked_hands = trackers.process(rgb)

        face_w = 0.28 * w  # fallback when no face is visible
        if face_pts is not None:
            face_pts = face_pts * np.float32([w, h])
            face_w = float(np.linalg.norm(face_pts[FACE_LEFT] - face_pts[FACE_RIGHT]))

        base_radius = max(40.0, 0.55 * face_w)
        snap_dist = 0.65 * face_w
        max_stretch = args.max_stretch * face_w

        grabs = []
        hand_pts_list = []
        seen_labels = set()

        for pts, label in tracked_hands:
            pts = pts * np.float32([w, h])
            info = hand_pinch_info(pts)
            if info is None:
                continue
            pinch_pt, pinch_ratio = info
            hand_pts_list.append(pts)
            while label in seen_labels:  # two hands can share a handedness
                label += "'"
            seen_labels.add(label)
            grab = grabs_by_hand.setdefault(label, HandGrab())

            if not grab.active and pinch_ratio < PINCH_ENGAGE:
                grab.engage(pinch_pt, face_pts, snap_dist)
            elif grab.active and pinch_ratio > PINCH_RELEASE:
                anchor = grab.anchor(face_pts)
                if anchor is not None and grab.q is not None:
                    snapbacks.append(SnapBack(anchor, grab.q, base_radius))
                grab.release()
            elif grab.active:
                grab.update(pinch_pt)

        # A hand that disappears from tracking mid-grab also snaps back.
        for label, grab in grabs_by_hand.items():
            if grab.active and label not in seen_labels:
                anchor = grab.anchor(face_pts)
                if anchor is not None and grab.q is not None:
                    snapbacks.append(SnapBack(anchor, grab.q, base_radius))
                grab.release()

        for grab in grabs_by_hand.values():
            if not grab.active:
                continue
            anchor = grab.anchor(face_pts)
            if anchor is None or grab.q is None:
                continue
            px, py = anchor
            qx, qy = grab.q
            vx, vy = qx - px, qy - py
            stretch = math.hypot(vx, vy)
            if stretch < 2.0:
                continue
            if stretch > max_stretch:  # rubber has limits, even Luffy's
                k = max_stretch / stretch
                qx, qy = px + vx * k, py + vy * k
            grabs.append(make_grab((px, py), (qx, qy), base_radius))

        snapbacks = [s for s in snapbacks if s.grab() is not None]
        for s in snapbacks:
            g = s.grab()
            if g is not None:
                grabs.append(g)

        out = apply_warp(frame, grabs)

        # Draw the real hands back on top so the fingers appear to hold the
        # stretched skin instead of being smeared by the warp.
        if grabs and hand_overlay and hand_pts_list:
            alpha = hand_overlay_mask(hand_pts_list, h, w)
            if alpha is not None:
                out = (frame.astype(np.float32) * alpha
                       + out.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        if debug:
            if face_pts is not None:
                for x, y in face_pts[::4]:
                    cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
            for px, py, qx, qy, r_behind, r_ahead, r_perp in grabs:
                cv2.circle(out, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.circle(out, (int(qx), int(qy)), 5, (255, 0, 0), -1)
                cv2.line(out, (int(px), int(py)), (int(qx), int(qy)),
                         (0, 255, 255), 2)
                angle = math.degrees(math.atan2(qy - py, qx - px))
                cv2.ellipse(out, ((int(qx), int(qy)),
                                  (int(2 * r_behind), int(2 * r_perp)), angle),
                            (255, 255, 0), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(out, "Pinch your cheek & pull  |  q quit  d debug",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, f"{fps:4.1f} fps", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Luffy Effect", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            debug = not debug
        elif key == ord('o'):
            hand_overlay = not hand_overlay
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()
    trackers.close()


# ----------------------------------------------------------------------------
# Self test: exercises the warp + snap-back math without camera or MediaPipe
# ----------------------------------------------------------------------------

def _draw_test_face(w=480, h=480):
    img = np.full((h, w, 3), 235, np.uint8)
    for x in range(0, w, 30):
        cv2.line(img, (x, 0), (x, h), (210, 210, 210), 1)
    for y in range(0, h, 30):
        cv2.line(img, (0, y), (w, y), (210, 210, 210), 1)
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy), 150, (90, 170, 240), -1)      # face
    cv2.circle(img, (cx - 55, cy - 45), 18, (60, 60, 60), -1)  # eyes
    cv2.circle(img, (cx + 55, cy - 45), 18, (60, 60, 60), -1)
    cv2.ellipse(img, (cx, cy + 45), (70, 40), 0, 15, 165, (50, 50, 50), 8)
    return img


def selftest(out_path):
    img = _draw_test_face()
    h, w = img.shape[:2]
    anchor = (w // 2 - 95, h // 2 + 45)  # left cheek / mouth corner

    # The grabbed pixel must land exactly at the pull point.
    pull = (60, h // 2 + 60)
    map_x, map_y = build_maps(h, w, [make_grab(anchor, pull, 120)], scale=1.0)
    sx, sy = map_x[pull[1], pull[0]], map_y[pull[1], pull[0]]
    assert abs(sx - anchor[0]) < 1.0 and abs(sy - anchor[1]) < 1.0, \
        f"grabbed pixel should land at the fingertip, got source ({sx:.1f},{sy:.1f})"

    # Pixels outside the radius must be untouched.
    far = (w - 5, 5)
    assert abs(map_x[far[1], far[0]] - far[0]) < 1e-3
    assert abs(map_y[far[1], far[0]] - far[1]) < 1e-3

    # Snap-back spring decays to rest.
    sb = SnapBack(anchor, pull, 120)
    sb.t0 -= 5.0  # pretend 5 seconds passed
    assert sb.grab() is None, "snap-back should settle"

    # Render a montage: idle, mid pull, full pull, overshoot wobble.
    def stretched(qx, qy):
        return apply_warp(img, [make_grab(anchor, (qx, qy), 120)])

    mid = ((anchor[0] + pull[0]) // 2, (anchor[1] + pull[1]) // 2)
    over = (anchor[0] + int((anchor[0] - pull[0]) * 0.25),
            anchor[1] + int((anchor[1] - pull[1]) * 0.25))
    panels = [img, stretched(*mid), stretched(*pull), stretched(*over)]
    labels = ["idle", "pulling", "hand stops here", "snap-back wobble"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (30, 30, 30), 2)
    montage = cv2.hconcat(panels)
    cv2.imwrite(out_path, montage)
    print(f"selftest OK -- montage written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Luffy rubber-stretch webcam effect")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-stretch", type=float, default=3.0,
                    help="max stretch length, in face-widths")
    ap.add_argument("--model-complexity", type=int, default=1, choices=(0, 1),
                    help="hand model complexity (0 is faster, 1 more accurate)")
    ap.add_argument("--selftest", nargs="?", const="luffy_selftest.png",
                    metavar="OUT.png",
                    help="run the warp self-test (no camera needed) and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.selftest)
        return
    run_live(args)


if __name__ == "__main__":
    main()
