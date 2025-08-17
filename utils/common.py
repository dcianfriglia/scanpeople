import math
import time
import numpy as np

# ---------- Geometry & helpers ----------

def dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def angle3(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (nba*nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def ema(prev, new, alpha=0.3):
    return alpha*new + (1-alpha)*prev if prev is not None else new

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

def to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

# Sane normalization anchors from the face mesh
FACEMESH_IDX = {
    # Mouth
    "MOUTH_LEFT": 61, "MOUTH_RIGHT": 291, "MOUTH_TOP_IN": 13, "MOUTH_BOTTOM_IN": 14,
    # Eyes (corners + lids)
    "R_EYE_OUT": 33, "R_EYE_IN": 133, "R_EYE_UP": 159, "R_EYE_DN": 145,
    "L_EYE_OUT": 362, "L_EYE_IN": 263, "L_EYE_UP": 386, "L_EYE_DN": 374,
    # Brows (inner points)
    "R_BROW_IN": 105, "L_BROW_IN": 334,
    # Nose tip & chin
    "NOSE_TIP": 1, "CHIN": 152,
}

class FPSCounter:
    def __init__(self):
        self.prev = time.time()
        self.fps = 0.0
    def tick(self):
        now = time.time()
        self.fps = 1.0 / max(1e-6, now - self.prev)
        self.prev = now
        return self.fps
