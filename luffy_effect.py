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
MAP_SCALE = 0.35
# Fraction of the flap half-width that moves rigidly (no shear smear inside).
PLATEAU = 0.40
# Flap half-width at the fingertip, as a fraction of the root half-width.
TIP_WIDTH = 0.35

# --- skin-flap overlay (all lengths are fractions of the face width) -------
FLAP_ROOT_BACK = 0.18   # flap begins this far behind the grab anchor
FLAP_SRC_LEN = 0.35     # how much real skin is fed into the stretch
FLAP_ROOT_HALF = 0.20   # half-width of the flap at its root
FLAP_TIP_MIN = 0.55     # tip half-width as a fraction of the root (minimum)
FLAP_TIP_CAP = 0.86     # where the rounded tip cap begins (fraction of length)
FLAP_FEATHER = 0.25     # feathered edge, as a fraction of the half-width
FLAP_SPEC = 0.10        # specular highlight strength on stretched skin
FLAP_BRIGHT = 0.08      # brightening of fully stretched skin
FLAP_SAG = 0.10         # gravity droop of the flap (fraction of its length)
FLAP_SHADE = 0.10       # top-lit shading: brighter upper edge, darker lower
# How hard the face itself is dragged toward the hand (the widened grin).
ROOT_PULL = 0.40        # fraction of the stretch
ROOT_PULL_MAX = 0.55    # cap, in face widths


def make_grab(anchor, pull, base_radius):
    """Build one grab tuple: pull the pixel at `anchor` to `pull`.

    The influence region is a cone-shaped flap: narrow at the fingertips,
    widening back to the skin it was grabbed from, with almost no reach
    past the fingertips so the rest of the frame stays put.
    """
    stretch = math.hypot(pull[0] - anchor[0], pull[1] - anchor[1])
    r_behind = stretch + 0.55 * base_radius  # flap root, just behind anchor
    r_ahead = max(24.0, 0.35 * base_radius)  # past the fingertip: stay short
    r_perp = max(24.0, 0.42 * base_radius + 0.08 * stretch)
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

        # Along the pull: a LINEAR ramp behind the fingertip. Its derivative
        # is constant, so the skin texture stretches uniformly from root to
        # tip -- a clean rubber stretch instead of a smeared blur. Because
        # r_behind = stretch + margin, the ramp never folds the image over.
        back = np.clip(1.0 + along / r_behind, 0.0, 1.0)
        ahead = np.clip(1.0 - (along / r_ahead) ** 2, 0.0, None)
        ahead *= ahead                 # quick smooth falloff past the tip
        k_along = np.where(along <= 0, back, ahead).astype(np.float32)

        # Across the pull: a cone, narrow at the fingertip and widening back
        # to the grabbed skin, with a rigid plateau core so the middle of the
        # flap moves as one piece and only a thin border band shears.
        width = np.where(along <= 0,
                         TIP_WIDTH + (1.0 - TIP_WIDTH) * (-along / r_behind),
                         np.float32(TIP_WIDTH)).astype(np.float32)
        tp = np.abs(perp) / (r_perp * width)
        s = np.clip((tp - PLATEAU) / (1.0 - PLATEAU), 0.0, 1.0)
        k = k_along * (1.0 - s * s * (3.0 - 2.0 * s))
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


def _flap_dims(stretch, scale, ny):
    """Shared flap geometry: (root_back, half_root, half_tip, sag).

    The sag is capped below the tip half-width so the drooped flap can
    never uncover the face-drag warp that runs underneath it."""
    root_back = FLAP_ROOT_BACK * scale
    half_root = max(12.0, FLAP_ROOT_HALF * scale)
    half_tip = half_root * max(FLAP_TIP_MIN, 1.0 - 0.30 * stretch / scale)
    sag = min(FLAP_SAG * stretch, 0.9 * half_tip) * ny
    return root_back, half_root, half_tip, sag


def root_grab(anchor, pull, scale):
    """Grab tuple that drags the face itself toward the hand -- this is what
    widens the grin and bares the teeth. The drag is aimed along the flap's
    drooped centerline so the warped cheek and the flap overlay line up.
    Returns None when the pull is too small to matter."""
    vx, vy = pull[0] - anchor[0], pull[1] - anchor[1]
    stretch = math.hypot(vx, vy)
    if stretch < 2.0:
        return None
    ux, uy = vx / stretch, vy / stretch
    nx, ny = -uy, ux
    k_len = min(ROOT_PULL * stretch, ROOT_PULL_MAX * scale)
    root_back, _, _, sag = _flap_dims(stretch, scale, ny)
    t_root = (root_back + k_len) / (root_back + stretch)
    bow = sag * 4.0 * t_root * (1.0 - t_root)
    target = (anchor[0] + ux * k_len + nx * bow,
              anchor[1] + uy * k_len + ny * bow)
    g = make_grab(anchor, target, 0.6 * scale)
    # The flap overlay covers everything past the drag target, so the warp
    # needs almost no forward reach of its own.
    return g[:5] + (max(16.0, 0.10 * scale),) + g[6:]


def render_flap(dst, src, anchor, pull, scale):
    """Draw a stretched-skin flap from `anchor` to `pull` on top of `dst`.

    This is what makes the effect clean: instead of liquify-warping the
    whole frame, the flap is an explicit textured strip -- real skin
    sampled around the anchor, stretched uniformly along its length, with
    feathered edges, a rounded tip, and a specular sheen. The background
    is never dragged, so there is no smearing. `src` supplies the clean
    skin texture (the unwarped camera frame); `scale` is the face width.
    """
    h, w = src.shape[:2]
    px, py = float(anchor[0]), float(anchor[1])
    qx, qy = float(pull[0]), float(pull[1])
    vx, vy = qx - px, qy - py
    stretch = math.hypot(vx, vy)
    if stretch < 2.0:
        return dst
    ux, uy = vx / stretch, vy / stretch
    nx, ny = -uy, ux

    # Heavy skin droops: the strip's centerline bows downward, zero at both
    # ends (root and fingers) and largest mid-flap.
    root_back, half_root, half_tip, sag = _flap_dims(stretch, scale, ny)
    ax, ay = px - ux * root_back, py - uy * root_back
    length = root_back + stretch           # strip reaches exactly the pinch
    src_len = root_back + min(stretch, FLAP_SRC_LEN * scale)
    gravity_n = ny                    # how vertical the strip's normal is

    span = half_root + abs(sag) + 1.0
    x0 = max(0, int(min(ax, qx) - span) - 4)
    x1 = min(w, int(max(ax, qx) + span) + 5)
    y0 = max(0, int(min(ay, qy) - span) - 4)
    y1 = min(h, int(max(ay, qy) + span) + 5)
    if x1 <= x0 or y1 <= y0:
        return dst

    X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                       np.arange(y0, y1, dtype=np.float32))
    t = ((X - ax) * ux + (Y - ay) * uy) / length   # 0 at root, 1 at pinch
    sdist = (X - ax) * nx + (Y - ay) * ny          # signed lateral distance
    tc = np.clip(t, 0.0, 1.0)
    sdist = sdist - sag * 4.0 * tc * (1.0 - tc)    # follow the drooped line
    wt = half_root + (half_tip - half_root) * tc
    s = sdist / np.maximum(wt, 1e-3)               # -1..1 across the flap

    # Rounded tip: past FLAP_TIP_CAP the allowed width shrinks like a circle.
    f = np.clip((t - FLAP_TIP_CAP) / (1.0 - FLAP_TIP_CAP), 0.0, 1.0)
    s_eff = np.abs(s) / np.maximum(np.sqrt(1.0 - f * f), 1e-4)

    inside = (t >= 0.0) & (t <= 1.0) & (s_eff <= 1.0)
    if not inside.any():
        return dst

    # Sample the real skin: linear along the strip (uniform stretch, no
    # smear); laterally always from the full root width, so the texture
    # pinches together toward the fingertips like held skin.
    map_x = ax + ux * (t * src_len) + nx * (s * half_root)
    map_y = ay + uy * (t * src_len) + ny * (s * half_root)
    flap = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

    # Shading: stretched rubber skin lightens and catches a specular streak
    # down its center; the feathered edges roll away and darken slightly.
    stretch_f = min(1.0, max(0.0, (length - src_len) / max(src_len, 1.0)))
    edge = np.clip((s_eff - (1.0 - FLAP_FEATHER)) / FLAP_FEATHER, 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    spec = (FLAP_SPEC * stretch_f
            * np.clip(1.0 - s_eff * s_eff, 0.0, 1.0) ** 2
            * np.clip(4.0 * t * (1.0 - t), 0.0, 1.0))
    # Top-lit cylinder shading: the edge facing up catches light, the one
    # facing down falls into shadow -- sells the roundness of the flap.
    shade = FLAP_SHADE * stretch_f * gravity_n * -np.clip(s, -1.0, 1.0)
    gain = (1.0 + FLAP_BRIGHT * stretch_f + spec + shade
            - 0.15 * stretch_f * edge)
    flap = np.clip(flap.astype(np.float32) * gain[..., None], 0, 255)

    ramp = np.clip(t / 0.12, 0.0, 1.0)     # blend out of the cheek at root
    alpha = np.where(inside, ramp * (1.0 - edge), 0.0).astype(np.float32)
    alpha = alpha[..., None]
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (flap * alpha + region * (1.0 - alpha)).astype(np.uint8)
    return dst


class SnapBack:
    """Damped spring that wobbles a released grab back to its anchor."""

    DECAY = 8.0        # 1/s exponential decay
    FREQ_HZ = 3.2      # wobble frequency
    DONE_BELOW = 0.02  # amplitude at which the wobble is considered finished

    def __init__(self, anchor, release_point):
        self.anchor = anchor
        self.v0 = (release_point[0] - anchor[0], release_point[1] - anchor[1])
        self.t0 = time.time()

    def pull(self):
        """Current (anchor, pull) pair for this wobble, or None once settled."""
        t = time.time() - self.t0
        s = math.exp(-t * self.DECAY) * math.cos(2 * math.pi * self.FREQ_HZ * t)
        if abs(s) < self.DONE_BELOW:
            return None
        px, py = self.anchor
        return ((px, py), (px + self.v0[0] * s, py + self.v0[1] * s))


# ----------------------------------------------------------------------------
# Face/hand tracking (MediaPipe imported lazily so --selftest works without it)
# ----------------------------------------------------------------------------

PINCH_ENGAGE = 0.32   # pinch when tip distance / hand size drops below this
PINCH_RELEASE = 0.50  # release when it rises above this (hysteresis)
ENGAGE_FRAMES = 2     # pinch must hold this many frames before grabbing
DEAD_ZONE = 0.06      # no warp until the hand moves this many face-widths
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
        self.ready = False      # hand must be seen OPEN before it can grab
        self.pinch_frames = 0
        self.face_idx = None    # face landmark the grab is glued to
        self.offset = (0.0, 0.0)  # pinch point relative to that landmark
        self.fixed_anchor = None  # anchor for grabs that start off the face
        self.q = None           # smoothed pull point (fingertips)

    def anchor(self, face_pts):
        if self.face_idx is not None and face_pts is not None:
            lx, ly = face_pts[self.face_idx]
            return (float(lx) + self.offset[0], float(ly) + self.offset[1])
        return self.fixed_anchor

    def engage(self, pinch_pt, face_pts, snap_dist):
        """Start a grab with ZERO displacement: the anchor is exactly the
        pinch point, so nothing moves until the hand actually pulls. Near
        the face the anchor is stored relative to the nearest face landmark
        so it stays glued to the cheek while the head moves."""
        self.active = True
        self.q = pinch_pt
        self.face_idx = None
        self.fixed_anchor = pinch_pt
        if face_pts is not None:
            d = np.linalg.norm(face_pts - np.float32(pinch_pt), axis=1)
            idx = int(np.argmin(d))
            if d[idx] <= snap_dist:
                self.face_idx = idx
                self.offset = (pinch_pt[0] - float(face_pts[idx][0]),
                               pinch_pt[1] - float(face_pts[idx][1]))
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


def effective_pull(anchor, q, dead_zone):
    """Pull point with the dead zone subtracted, or None if the hand hasn't
    moved far enough from the anchor for the warp to kick in."""
    vx, vy = q[0] - anchor[0], q[1] - anchor[1]
    stretch = math.hypot(vx, vy)
    if stretch <= dead_zone + 1.0:
        return None
    k = (stretch - dead_zone) / stretch
    return (anchor[0] + vx * k, anchor[1] + vy * k)


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

        snap_dist = 0.65 * face_w
        max_stretch = args.max_stretch * face_w
        dead_zone = DEAD_ZONE * face_w

        pulls = []   # (anchor, fingertip) pairs to render this frame
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

            if not grab.active:
                # Require an open hand first, then a held pinch: this stops
                # accidental grabs while the hand is still approaching the
                # face (which used to shove pixels around before the grab).
                if pinch_ratio > PINCH_RELEASE:
                    grab.ready = True
                    grab.pinch_frames = 0
                elif grab.ready and pinch_ratio < PINCH_ENGAGE:
                    grab.pinch_frames += 1
                    if grab.pinch_frames >= ENGAGE_FRAMES:
                        grab.engage(pinch_pt, face_pts, snap_dist)
                        grab.ready = False
                        grab.pinch_frames = 0
            elif pinch_ratio > PINCH_RELEASE:
                anchor = grab.anchor(face_pts)
                if anchor is not None and grab.q is not None:
                    q_eff = effective_pull(anchor, grab.q, dead_zone)
                    if q_eff is not None:
                        snapbacks.append(SnapBack(anchor, q_eff))
                grab.release()
            else:
                grab.update(pinch_pt)

        # A hand that disappears from tracking mid-grab also snaps back.
        for label, grab in grabs_by_hand.items():
            if grab.active and label not in seen_labels:
                anchor = grab.anchor(face_pts)
                if anchor is not None and grab.q is not None:
                    q_eff = effective_pull(anchor, grab.q, dead_zone)
                    if q_eff is not None:
                        snapbacks.append(SnapBack(anchor, q_eff))
                grab.release()

        for grab in grabs_by_hand.values():
            if not grab.active:
                continue
            anchor = grab.anchor(face_pts)
            if anchor is None or grab.q is None:
                continue
            q_eff = effective_pull(anchor, grab.q, dead_zone)
            if q_eff is None:
                continue
            px, py = anchor
            qx, qy = q_eff
            vx, vy = qx - px, qy - py
            stretch = math.hypot(vx, vy)
            if stretch > max_stretch:  # rubber has limits, even Luffy's
                k = max_stretch / stretch
                qx, qy = px + vx * k, py + vy * k
            pulls.append(((px, py), (qx, qy)))

        alive = []
        for sb in snapbacks:
            pq = sb.pull()
            if pq is not None:
                alive.append(sb)
                pulls.append(pq)
        snapbacks = alive

        # Drag the face itself toward the hand (this is what widens the grin
        # and bares the teeth in the real video); the flap continues from
        # the dragged cheek as a clean overlay.
        base_grabs = []
        for p, q in pulls:
            g = root_grab(p, q, face_w)
            if g is not None:
                base_grabs.append(g)

        out = apply_warp(frame, base_grabs)
        if out is frame and pulls:
            out = frame.copy()
        for p, q in pulls:
            out = render_flap(out, frame, p, q, face_w)

        # Draw the real hands back on top so the fingers appear to hold the
        # stretched skin instead of being covered by it.
        if pulls and hand_overlay and hand_pts_list:
            alpha = hand_overlay_mask(hand_pts_list, h, w)
            if alpha is not None:
                out = (frame.astype(np.float32) * alpha
                       + out.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        if debug:
            if face_pts is not None:
                for x, y in face_pts[::4]:
                    cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
            for (px, py), (qx, qy) in pulls:
                cv2.circle(out, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.circle(out, (int(qx), int(qy)), 5, (255, 0, 0), -1)
                cv2.line(out, (int(px), int(py)), (int(qx), int(qy)),
                         (0, 255, 255), 2)

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
    sb = SnapBack(anchor, pull)
    sb.t0 -= 5.0  # pretend 5 seconds passed
    assert sb.pull() is None, "snap-back should settle"

    scale = 200.0  # test face is ~300 px wide; scale plays the face width

    # A tiny stretch must leave the frame untouched (no push on pinch).
    tiny = render_flap(img.copy(), img, anchor, (anchor[0] + 1, anchor[1]), scale)
    assert np.array_equal(tiny, img), "flap must be invisible before pulling"

    # The flap must stay inside its own corridor between anchor and pull.
    flapped = render_flap(img.copy(), img, anchor, pull, scale)
    diff = np.abs(flapped.astype(np.int16) - img.astype(np.int16)).max(axis=2)
    ys, xs = np.nonzero(diff > 8)
    assert xs.size, "flap should be visible at full stretch"
    ux, uy = np.float32(pull) - np.float32(anchor)
    seg = math.hypot(ux, uy)
    ux, uy = ux / seg, uy / seg
    along = (xs - anchor[0]) * ux + (ys - anchor[1]) * uy
    perp = np.abs((ys - anchor[1]) * ux - (xs - anchor[0]) * uy)
    margin = FLAP_ROOT_HALF * scale + FLAP_SAG * seg + 6
    assert along.min() > -FLAP_ROOT_BACK * scale - 6 and along.max() < seg + 6, \
        "flap leaked along the pull axis"
    assert perp.max() < margin, "flap leaked sideways"

    # Render a montage: idle, mid pull, full pull, overshoot wobble.
    def stretched(qx, qy):
        g = root_grab(anchor, (qx, qy), scale)
        if g is None:
            return img.copy()
        out = apply_warp(img, [g])
        return render_flap(out, img, anchor, (qx, qy), scale)

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
