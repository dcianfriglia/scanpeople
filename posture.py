#!/usr/bin/env python3
"""
Posture & Walking-Style Labeller
- YOLOv8-Pose + simple tracker + heuristic labels
- Self-scaling writer (prevents cropping)
- Interactive viewer:
    * CTRL + mouse wheel = zoom in/out (anchored at cursor)  [viewer only]
    * Shift + Left-drag OR Middle-drag = pan                 [viewer only]
    * Click ID in sidebar = toggle body lines on/off
    * Double-click a person's outline in video = pin/unpin ID at top of sidebar
    * R = reset zoom/pan
    * ESC/Q = quit
- Sidebar with live list of detected people and labels (viewer only)
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Literal, NamedTuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is required. Install with: pip install ultralytics") from e


# -----------------------------
# Types
# -----------------------------

Category = Literal["posture", "walking"]


class Label(NamedTuple):
    category: Category
    label: str
    confidence: float


Point = Tuple[float, float]


# COCO keypoint indices for YOLOv8-Pose
KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_wri": 9,
    "r_wri": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ank": 15,
    "r_ank": 16,
}

SKELETON_EDGES = [
    (KP["l_sho"], KP["r_sho"]),
    (KP["l_sho"], KP["l_elb"]),
    (KP["l_elb"], KP["l_wri"]),
    (KP["r_sho"], KP["r_elb"]),
    (KP["r_elb"], KP["r_wri"]),
    (KP["l_sho"], KP["l_hip"]),
    (KP["r_sho"], KP["r_hip"]),
    (KP["l_hip"], KP["r_hip"]),
    (KP["l_hip"], KP["l_knee"]),
    (KP["l_knee"], KP["l_ank"]),
    (KP["r_hip"], KP["r_knee"]),
    (KP["r_knee"], KP["r_ank"]),
    (KP["nose"], KP["l_sho"]),
    (KP["nose"], KP["r_sho"]),
]


# -----------------------------
# Utility math (strict scalars)
# -----------------------------

def _to_py_float(x: float | int | np.ndarray) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    return float(np.asarray(x).reshape(-1)[0])


def _to_py_int(x: float | int | np.ndarray) -> int:
    return int(round(_to_py_float(x)))


def dist(a: Point | None, b: Point | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def angle_deg(vec: Tuple[float, float]) -> float:
    return float(math.degrees(math.atan2(vec[1], vec[0])))


def safe_pt(kps: np.ndarray, idx: int) -> Optional[Point]:
    if idx < 0 or idx >= kps.shape[0]:
        return None
    x, y, c = kps[idx]
    if _to_py_float(c) < 0.2:
        return None
    return (float(x), float(y))


def mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


# -----------------------------
# Lightweight tracker
# -----------------------------

@dataclass
class Track:
    tid: int
    last_center: Point
    age: int = 0
    misses: int = 0
    centers: Deque[Point] = field(default_factory=lambda: deque(maxlen=30))
    ankle_y: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))  # (l_ank_y, r_ank_y)
    kps_hist: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))

    def update(self, center: Point, kps: np.ndarray, l_ank_y: float, r_ank_y: float) -> None:
        self.age += 1
        self.misses = 0
        self.last_center = center
        self.centers.append(center)
        self.kps_hist.append(kps)
        self.ankle_y.append((float(l_ank_y), float(r_ank_y)))

    def mark_missed(self) -> None:
        self.age += 1
        self.misses += 1


class GreedyTracker:
    def __init__(self, max_misses: int = 30, max_dist_px: float = 120.0) -> None:
        self.max_misses = max_misses
        self.max_dist_px = max_dist_px
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, center: Point, kps: np.ndarray, l_ank_y: float, r_ank_y: float) -> Track:
        t = Track(tid=self._next_id, last_center=center)
        t.update(center, kps, l_ank_y, r_ank_y)
        self.tracks[t.tid] = t
        self._next_id += 1
        return t

    def update(self, detections: List[Tuple[Point, np.ndarray]]) -> Dict[int, Track]:
        det_used = [False] * len(detections)

        # Match existing tracks
        for tid, tr in list(self.tracks.items()):
            best_j = -1
            best_d = 1e9
            for j, (center, _) in enumerate(detections):
                if det_used[j]:
                    continue
                d = dist(tr.last_center, center)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d <= self.max_dist_px:
                c, kps = detections[best_j]
                l_ank_y = float(kps[KP["l_ank"], 1])
                r_ank_y = float(kps[KP["r_ank"], 1])
                tr.update(c, kps, l_ank_y, r_ank_y)
                det_used[best_j] = True
            else:
                tr.mark_missed()
                if tr.misses > self.max_misses:
                    del self.tracks[tid]

        # New tracks
        for j, used in enumerate(det_used):
            if not used:
                c, kps = detections[j]
                l_ank_y = float(kps[KP["l_ank"], 1])
                r_ank_y = float(kps[KP["r_ank"], 1])
                self._new_track(c, kps, l_ank_y, r_ank_y)

        return self.tracks


# -----------------------------
# Heuristic classifier
# -----------------------------

class HeuristicClassifier:
    def classify(self, tr: Track) -> Tuple[Label, Label]:
        posture = Label("posture", "Open Stance", 0.50)
        walking = Label("walking", "Casual Stroll", 0.50)

        if not tr.kps_hist:
            return posture, walking

        kps = tr.kps_hist[-1]

        # Points
        l_sho = safe_pt(kps, KP["l_sho"])
        r_sho = safe_pt(kps, KP["r_sho"])
        l_hip = safe_pt(kps, KP["l_hip"])
        r_hip = safe_pt(kps, KP["r_hip"])
        l_wri = safe_pt(kps, KP["l_wri"])
        r_wri = safe_pt(kps, KP["r_wri"])
        l_elb = safe_pt(kps, KP["l_elb"])
        r_elb = safe_pt(kps, KP["r_elb"])
        l_ank = safe_pt(kps, KP["l_ank"])
        r_ank = safe_pt(kps, KP["r_ank"])
        nose  = safe_pt(kps, KP["nose"])
        l_ear = safe_pt(kps, KP["l_ear"])
        r_ear = safe_pt(kps, KP["r_ear"])

        shoulder_w = dist(l_sho, r_sho) if (l_sho and r_sho) else 0.0
        hip_mid: Optional[Point] = mid(l_hip, r_hip) if (l_hip and r_hip) else None

        # Torso lean
        torso_angle = 0.0
        if l_sho and r_sho and hip_mid:
            shoulder_mid = mid(l_sho, r_sho)
            v = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
            angle_from_vertical = abs(angle_deg(v) - (-90.0))
            torso_angle = float(min(angle_from_vertical, 180.0 - angle_from_vertical))

        # Stance width
        stance_ratio = dist(l_ank, r_ank) / max(shoulder_w, 1e-3) if (l_ank and r_ank and shoulder_w > 1e-3) else 0.0

        # Arms crossed
        arms_crossed = False
        if l_wri and r_wri and l_elb and r_elb and shoulder_w > 1e-3 and l_sho and r_sho:
            chest = mid(l_sho, r_sho)
            wdist = dist(l_wri, r_wri)
            if wdist < 0.9 * shoulder_w and dist(l_wri, chest) < 1.2 * shoulder_w and dist(r_wri, chest) < 1.2 * shoulder_w:
                arms_crossed = True

        # Hands on hips
        hands_on_hips = False
        if l_wri and r_wri and l_hip and r_hip and shoulder_w > 1e-3:
            if dist(l_wri, l_hip) < 0.7 * shoulder_w and dist(r_wri, r_hip) < 0.7 * shoulder_w:
                hands_on_hips = True

        # Head down
        head_down = False
        if nose and l_sho and r_sho:
            shoulder_mid = mid(l_sho, r_sho)
            head_down = (nose[1] - shoulder_mid[1]) > 0.25 * max(shoulder_w, 1.0)

        # Speed (px/frame)
        speed = 0.0
        if len(tr.centers) >= 2:
            dsum = 0.0
            for i in range(1, len(tr.centers)):
                a, b = tr.centers[i - 1], tr.centers[i]
                dsum += dist(a, b)
            speed = dsum / float(len(tr.centers) - 1)

        # Step asymmetry
        step_asym = 0.0
        if len(tr.ankle_y) >= 6:
            diffs = [abs(a - b) for (a, b) in tr.ankle_y]
            m = float(sum(diffs) / len(diffs))
            var = float(sum((d - m) ** 2 for d in diffs) / len(diffs))
            step_asym = var ** 0.5

        # Phone walk
        phone_like = False
        if (l_wri or r_wri) and (l_ear or r_ear) and shoulder_w > 1e-3 and speed > 1.5:
            d_candidates: List[float] = []
            if l_wri and l_ear:
                d_candidates.append(dist(l_wri, l_ear))
            if r_wri and r_ear:
                d_candidates.append(dist(r_wri, r_ear))
            if d_candidates:
                phone_like = (min(d_candidates) / shoulder_w) < 0.6

        # Posture
        posture_candidates: List[Tuple[str, float]] = []
        if stance_ratio > 1.8:
            posture_candidates.append(("Wide Stance", min(1.0, (stance_ratio - 1.8) / 1.0)))
        if arms_crossed:
            posture_candidates.append(("Arms Crossed", 0.9))
        if hands_on_hips:
            posture_candidates.append(("Hands on Hips", 0.85))
        if head_down:
            posture_candidates.append(("Head Down", 0.75))
        if torso_angle > 15.0 and speed <= 1.5:
            posture_candidates.append(("Forward Lean", min(1.0, (torso_angle - 15.0) / 15.0 + 0.5)))
        if not posture_candidates:
            posture_candidates.append(("Open Stance", min(0.95, 0.55 + 0.15 * max(0.0, stance_ratio - 1.0))))
        best_posture = max(posture_candidates, key=lambda x: x[1])
        posture = Label("posture", best_posture[0], float(best_posture[1]))

        # Walking
        norm_speed = speed / (shoulder_w + 1e-6) if shoulder_w > 0 else speed
        walking_candidates: List[Tuple[str, float]] = []
        if norm_speed < 0.05:
            walking_candidates.append(("Standing", 0.9))
        else:
            if norm_speed < 0.15:
                walking_candidates.append(("Casual Stroll", 0.6))
            elif norm_speed < 0.35:
                walking_candidates.append(("Power Walk", 0.75))
            else:
                walking_candidates.append(("Charging Walk", 0.85))
            if phone_like:
                walking_candidates.append(("Phone Walk", 0.9))
            if norm_speed < 0.2 and step_asym > 3.0:
                walking_candidates.append(("Hesitant Steps", 0.8))
            if step_asym > 6.0:
                walking_candidates.append(("Limp", min(1.0, 0.7 + (step_asym - 6.0) / 6.0)))
            if norm_speed < 0.15 and step_asym < 1.5:
                walking_candidates.append(("Shuffle", 0.75))
        best_walk = max(walking_candidates, key=lambda x: x[1]) if walking_candidates else ("Standing", 0.8)
        walking = Label("walking", best_walk[0], float(best_walk[1]))

        return posture, walking


# -----------------------------
# Drawing helpers
# -----------------------------

def id_color(tid: int) -> Tuple[int, int, int]:
    rng = (tid * 123457) % 0xFFFFFF
    b = 50 + (rng & 0xFF) // 2
    g = 50 + ((rng >> 8) & 0xFF) // 2
    r = 50 + ((rng >> 16) & 0xFF) // 2
    return (int(b), int(g), int(r))


def draw_skeleton(frame: np.ndarray, kps: np.ndarray) -> None:
    for i, j in SKELETON_EDGES:
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if float(ci) < 0.2 or float(cj) < 0.2:
            continue
        cv2.line(frame, (_to_py_int(xi), _to_py_int(yi)), (_to_py_int(xj), _to_py_int(yj)), (0, 255, 0), 2)
    for x, y, c in kps:
        if float(c) < 0.2:
            continue
        cv2.circle(frame, (_to_py_int(x), _to_py_int(y)), 3, (255, 0, 0), -1)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow: bool = True,
) -> None:
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(thickness + 2), cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), color, int(thickness), cv2.LINE_AA)


# -----------------------------
# Self-scaling + Zoom/Pan viewer
# -----------------------------

def calc_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    return min(max_w / float(w), max_h / float(h), 1.0)


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ViewerState:
    def __init__(self) -> None:
        self.zoom: float = 1.0
        self.tx: float = 0.0  # translation in pixels (screen space)
        self.ty: float = 0.0
        self.mouse: Tuple[int, int] = (0, 0)
        self.dragging: bool = False
        self.drag_start: Tuple[int, int] = (0, 0)
        self.start_tx: float = 0.0
        self.start_ty: float = 0.0
        self.content_w: int = 0  # width of video content (excludes sidebar)

    def reset(self) -> None:
        self.zoom = 1.0
        self.tx = 0.0
        self.ty = 0.0
        self.dragging = False


@dataclass
class UIContext:
    viewer: ViewerState
    hidden_lines: set = field(default_factory=set)               # tids with skeleton hidden
    pinned_tid: Optional[int] = None                             # tid pinned to top of list
    sidebar_hitboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)  # tid -> (x1,y1,x2,y2) in window coords
    disp_bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)       # tid -> (x1,y1,x2,y2) in content coords
    sidebar_w: int = 300

    def toggle_lines(self, tid: int) -> None:
        if tid in self.hidden_lines:
            self.hidden_lines.remove(tid)
        else:
            self.hidden_lines.add(tid)

    def pin_or_unpin(self, tid: int) -> None:
        if self.pinned_tid == tid:
            self.pinned_tid = None
        else:
            self.pinned_tid = tid


def _get_wheel_delta(flags: int) -> int:
    if hasattr(cv2, "getMouseWheelDelta"):
        return int(cv2.getMouseWheelDelta(flags))
    return 120 if flags > 0 else -120


def make_mouse_callback(ctx: UIContext):
    v = ctx.viewer

    def cb(event: int, x: int, y: int, flags: int, userdata) -> None:
        x_i, y_i = int(x), int(y)
        v.mouse = (x_i, y_i)

        over_sidebar = (v.content_w > 0 and x_i >= v.content_w)

        if event == cv2.EVENT_MOUSEWHEEL:
            if over_sidebar:
                return
            ctrl = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
            if not ctrl:
                return
            delta = _get_wheel_delta(flags)
            steps = max(-10, min(10, int(round(delta / 120.0))))
            if steps == 0:
                return
            old_zoom = v.zoom
            factor = 1.2 ** steps
            new_zoom = max(0.25, min(8.0, old_zoom * factor))

            mx, my = x_i, y_i
            wx = (mx - v.tx) / max(old_zoom, 1e-6)
            wy = (my - v.ty) / max(old_zoom, 1e-6)
            v.tx = mx - wx * new_zoom
            v.ty = my - wy * new_zoom
            v.zoom = new_zoom

        elif event == cv2.EVENT_LBUTTONDOWN:
            if over_sidebar:
                # Click in sidebar: toggle lines if clicked on any ID row
                for tid, (sx1, sy1, sx2, sy2) in ctx.sidebar_hitboxes.items():
                    if sx1 <= x_i < sx2 and sy1 <= y_i < sy2:
                        ctx.toggle_lines(tid)
                        break
            else:
                # Start pan only if SHIFT is held
                if (flags & cv2.EVENT_FLAG_SHIFTKEY):
                    v.dragging = True
                    v.drag_start = (x_i, y_i)
                    v.start_tx = v.tx
                    v.start_ty = v.ty

        elif event == cv2.EVENT_MBUTTONDOWN:
            if not over_sidebar:
                v.dragging = True
                v.drag_start = (x_i, y_i)
                v.start_tx = v.tx
                v.start_ty = v.ty

        elif event == cv2.EVENT_MOUSEMOVE:
            if v.dragging:
                dx = x_i - v.drag_start[0]
                dy = y_i - v.drag_start[1]
                v.tx = v.start_tx + dx
                v.ty = v.start_ty + dy

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP):
            v.dragging = False

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if not over_sidebar:
                # Double-click in video: pin/unpin the ID whose outline contains the point
                for tid, (bx1, by1, bx2, by2) in ctx.disp_bboxes.items():
                    if bx1 <= x_i < bx2 and by1 <= y_i < by2:
                        ctx.pin_or_unpin(tid)
                        break

    return cb


def apply_view_transform(frame: np.ndarray, v: ViewerState) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.array([[v.zoom, 0.0, v.tx],
                  [0.0, v.zoom, v.ty]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# -----------------------------
# Sidebar UI
# -----------------------------

def render_sidebar(height: int,
                   people: List[Tuple[int, str, float, str, float]],
                   hidden_lines: set,
                   pinned_tid: Optional[int],
                   fps: float,
                   zoom: float,
                   panel_w: int,
                   base_x: int) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]]]:
    """
    Returns (panel_image, hitboxes). hitboxes map tid -> global window coords (x1,y1,x2,y2)
    Click inside an ID row toggles lines for that person.
    """
    panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    hitboxes: Dict[int, Tuple[int, int, int, int]] = {}

    y = 14
    def line(txt: str, color=(220, 220, 220), fs=0.5, thick=1):
        nonlocal y
        y += 16
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

    # Header
    cv2.rectangle(panel, (0, 0), (panel_w - 1, 40), (40, 40, 40), -1)
    cv2.putText(panel, "Detected People", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Stats
    y = 38
    line(f"Count: {len(people)}")
    line(f"FPS: {fps:.1f}")
    line(f"Zoom: {zoom:.2f}x")
    y += 6
    cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)

    # Instructions
    y += 8
    line("Controls:", (180, 200, 255))
    line("CTRL + wheel: Zoom", (180, 200, 255))
    line("Shift + L-drag: Pan", (180, 200, 255))
    line("Middle drag: Pan", (180, 200, 255))
    line("Click ID: Toggle lines", (180, 200, 255))
    line("Double-click outline: Pin", (180, 200, 255))
    y += 6
    cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)

    # List entries
    y += 8
    max_entries = max(1, (height - y - 10) // 60)
    # Reorder: pinned first if present
    if pinned_tid is not None:
        people = sorted(people, key=lambda x: (0 if x[0] == pinned_tid else 1, x[0]))
    else:
        people = sorted(people, key=lambda x: x[0])

    for tid, pl, pc, wl, wc in people[:max_entries]:
        entry_top = y
        entry_bottom = y + 56
        # Highlight pinned entry
        if tid == pinned_tid:
            cv2.rectangle(panel, (4, entry_top), (panel_w - 4, entry_bottom), (55, 55, 85), -1)
            cv2.putText(panel, "[PIN]", (panel_w - 70, entry_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 255), 1, cv2.LINE_AA)

        color = id_color(tid)
        bullet_color = (120, 120, 120) if tid in hidden_lines else color

        # Bullet + ID (click zone is the entire entry box)
        cv2.circle(panel, (18, y + 18), 6, bullet_color, -1)
        cv2.putText(panel, f"ID {tid}", (34, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

        # State hint
        if tid in hidden_lines:
            cv2.putText(panel, "[lines off]", (120, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 180), 1, cv2.LINE_AA)

        # Labels
        cv2.putText(panel, f"Posture: {pl} ({pc:.2f})", (14, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Walking: {wl} ({wc:.2f})", (14, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

        # Record hitbox in global window coords (content is on the left)
        sx1 = base_x
        sy1 = entry_top
        sx2 = base_x + panel_w
        sy2 = entry_bottom
        hitboxes[tid] = (sx1, sy1, sx2, sy2)

        y += 62

    return panel, hitboxes


# -----------------------------
# Main app
# -----------------------------

def run(
    source: str,
    save_out: Optional[str],
    device: str,
    max_width: int,
    max_height: int,
    show: bool,
) -> None:
    model = YOLO("yolov8n-pose.pt")

    cap: cv2.VideoCapture
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    writer: Optional[cv2.VideoWriter] = None
    tracker = GreedyTracker(max_misses=30, max_dist_px=160.0)
    classifier = HeuristicClassifier()

    win_name = "Posture & Gait Labeller"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    viewer = ViewerState()
    ctx = UIContext(viewer=viewer, sidebar_w=300)
    if show:
        cv2.setMouseCallback(win_name, make_mouse_callback(ctx), None)

    t_prev = time.time()
    fps = 0.0
    SIDEBAR_W = ctx.sidebar_w

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Timing for FPS
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)
        t_prev = t_now

        # Inference
        results = model(frame, verbose=False)[0]

        detections: List[Tuple[Point, np.ndarray]] = []
        if results.keypoints is not None and results.boxes is not None:
            kps_xy = results.keypoints.xy.cpu().numpy()  # (N,17,2)
            kps_conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else np.ones((kps_xy.shape[0], kps_xy.shape[1]), dtype=np.float32)
            kps = np.dstack([kps_xy, kps_conf])  # (N,17,3)

            for i in range(kps.shape[0]):
                if i < len(results.boxes):
                    box = results.boxes[i].xyxy.cpu().numpy().reshape(-1)
                    x1, y1, x2, y2 = map(float, box[:4])
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                else:
                    l_sho = safe_pt(kps[i], KP["l_sho"])
                    r_sho = safe_pt(kps[i], KP["r_sho"])
                    center = mid(l_sho, r_sho) if (l_sho and r_sho) else (float(kps[i, 0, 0]), float(kps[i, 0, 1]))
                detections.append((center, kps[i]))

        tracks = tracker.update(detections)

        # Draw overlays on original frame
        people_for_sidebar: List[Tuple[int, str, float, str, float]] = []
        raw_bboxes: Dict[int, Tuple[int, int, int, int]] = {}  # original-frame coords

        for tid, tr in tracks.items():
            if tr.misses > 0:
                continue
            kps_latest = tr.kps_hist[-1]

            # Bounding box from keypoints
            xs = kps_latest[:, 0]
            ys = kps_latest[:, 1]
            x1 = _to_py_int(np.nanmin(xs))
            y1 = _to_py_int(np.nanmin(ys))
            x2 = _to_py_int(np.nanmax(xs))
            y2 = _to_py_int(np.nanmax(ys))
            raw_bboxes[tid] = (x1, y1, x2, y2)

            # Draw skeleton only if not hidden
            if tid not in ctx.hidden_lines:
                draw_skeleton(frame, kps_latest)

            # Always draw a box and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), id_color(tid), 2)
            posture, walking = classifier.classify(tr)
            put_text(frame, f"ID {tid}", (x1, max(0, y1 - 10)), font_scale=0.6, color=(255, 255, 0))
            put_text(frame, f"{posture.category}: {posture.label} ({posture.confidence:.2f})", (x1, y1 + 18))
            put_text(frame, f"{walking.category}: {walking.label} ({walking.confidence:.2f})", (x1, y1 + 38))

            people_for_sidebar.append((tid, posture.label, posture.confidence, walking.label, walking.confidence))

        # -------- Self-scaling for writer (prevents cropping) --------
        h0, w0 = frame.shape[:2]
        scale = calc_scale(w0, h0, max_width, max_height)
        scaled_for_writer = scale_frame(frame, scale)

        # Create writer lazily AFTER we know final frame size
        if save_out and writer is None:
            h_out, w_out = scaled_for_writer.shape[:2]
            ext = save_out.lower().rsplit(".", 1)[-1] if "." in save_out else "mp4"
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext in ("mp4", "m4v", "mov", "mp4") else "XVID"))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            if vfps <= 1.0 or math.isnan(vfps):
                vfps = 25.0
            writer = cv2.VideoWriter(save_out, fourcc, float(vfps), (int(w_out), int(h_out)), True)

        if writer is not None:
            writer.write(scaled_for_writer)

        # -------- Interactive view: zoom/pan on video content ONLY --------
        viewer.content_w = int(scaled_for_writer.shape[1])  # so mouse callback can ignore sidebar area

        # Compute display bboxes (after scale + zoom/pan) for hit-testing dblclick
        ctx.disp_bboxes.clear()
        for tid, (x1, y1, x2, y2) in raw_bboxes.items():
            x1s, y1s = x1 * scale, y1 * scale
            x2s, y2s = x2 * scale, y2 * scale
            # apply viewer transform
            x1d = int(round(x1s * viewer.zoom + viewer.tx))
            y1d = int(round(y1s * viewer.zoom + viewer.ty))
            x2d = int(round(x2s * viewer.zoom + viewer.tx))
            y2d = int(round(y2s * viewer.zoom + viewer.ty))
            if x1d > x2d:
                x1d, x2d = x2d, x1d
            if y1d > y2d:
                y1d, y2d = y2d, y1d
            ctx.disp_bboxes[tid] = (x1d, y1d, x2d, y2d)

        content_display = apply_view_transform(scaled_for_writer, viewer)

        # Sidebar (viewer-only UI)
        sidebar, hitboxes = render_sidebar(
            height=content_display.shape[0],
            people=people_for_sidebar,
            hidden_lines=ctx.hidden_lines,
            pinned_tid=ctx.pinned_tid,
            fps=fps,
            zoom=viewer.zoom,
            panel_w=SIDEBAR_W,
            base_x=viewer.content_w,  # global window x offset for sidebar
        )
        ctx.sidebar_hitboxes = hitboxes  # update for mouse callback

        # Compose: content (left) + sidebar (right)
        display_frame = np.hstack([content_display, sidebar])

        if show:
            cv2.imshow(win_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                viewer.reset()

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to video file")
    ap.add_argument("--save-out", type=str, default="", help="Optional output video path (e.g., labeled.mp4)")
    ap.add_argument("--device", type=str, default="auto", help="Unused (ultralytics auto-selects); kept for compatibility")
    ap.add_argument("--max-width", type=int, default=1280, help="Max output width for scaling (prevents cropping)")
    ap.add_argument("--max-height", type=int, default=720, help="Max output height for scaling (prevents cropping)")
    ap.add_argument("--no-show", action="store_true", help="Disable display window")
    args = ap.parse_args()

    save_out = args.save_out if args.save_out.strip() else None
    run(
        source=args.source,
        save_out=save_out,
        device=args.device,
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()
