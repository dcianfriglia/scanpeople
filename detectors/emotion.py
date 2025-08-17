import time
import numpy as np
from utils.common import dist, ema, FACEMESH_IDX, clamp

EMOTIONS = ["happy","surprise","anger","sad","fear","disgust"]

class EmotionHeuristics:
    """
    Rule-based facial expression reader using MediaPipe FaceMesh indices.
    Supports neutral baseline (auto or press 'b'), per-emotion prototypes,
    and value/score debug readout.
    Emits: happy, surprise, anger, sad, fear, disgust, neutral.
    """
    def __init__(self):
        self.baseline = None            # neutral feature snapshot
        self.protos = {}               # per-emotion feature snapshots
        self.smooth_scores = {k: None for k in EMOTIONS}
        self.last_label = "neutral"
        self.last_scores = {k: 0.0 for k in EMOTIONS}
        self.last_ft = None            # last raw features
        self.last_d = None             # last deltas vs neutral
        self._boot_at = None

    # ---------- feature extraction ----------
    def _pt(self, lm, i, w, h):
        return (lm[i].x*w, lm[i].y*h)

    def _features(self, lm, w, h):
        f = FACEMESH_IDX
        L = self._pt(lm, f["MOUTH_LEFT"], w, h)
        R = self._pt(lm, f["MOUTH_RIGHT"], w, h)
        T = self._pt(lm, f["MOUTH_TOP_IN"], w, h)
        B = self._pt(lm, f["MOUTH_BOTTOM_IN"], w, h)
        mouth_w = dist(L, R)
        mouth_h = dist(T, B)
        MAR = mouth_h / (mouth_w + 1e-6)
        mouth_center_y = 0.5*(T[1] + B[1])
        smile_curve = (mouth_center_y - 0.5*(L[1]+R[1]))  # pixels; normalize by face scale

        r_eye_w = dist(self._pt(lm, f["R_EYE_OUT"], w, h), self._pt(lm, f["R_EYE_IN"], w, h))
        l_eye_w = dist(self._pt(lm, f["L_EYE_OUT"], w, h), self._pt(lm, f["L_EYE_IN"], w, h))
        r_eye_h = dist(self._pt(lm, f["R_EYE_UP"], w, h), self._pt(lm, f["R_EYE_DN"], w, h))
        l_eye_h = dist(self._pt(lm, f["L_EYE_UP"], w, h), self._pt(lm, f["L_EYE_DN"], w, h))
        eye_open = 0.5*((r_eye_h/(r_eye_w+1e-6)) + (l_eye_h/(l_eye_w+1e-6)))

        brow_gap = dist(self._pt(lm, f["R_BROW_IN"], w, h), self._pt(lm, f["L_BROW_IN"], w, h))
        rb_to_eye = abs(self._pt(lm, f["R_BROW_IN"], w, h)[1] - self._pt(lm, f["R_EYE_UP"], w, h)[1])
        lb_to_eye = abs(self._pt(lm, f["L_BROW_IN"], w, h)[1] - self._pt(lm, f["L_EYE_UP"], w, h)[1])
        nose_chin = dist(self._pt(lm, f["NOSE_TIP"], w, h), self._pt(lm, f["CHIN"], w, h))
        brow_raise = (rb_to_eye + lb_to_eye) / (2.0*(nose_chin+1e-6))
        nose_upperlip = abs(self._pt(lm, f["NOSE_TIP"], w, h)[1] - T[1]) / (nose_chin+1e-6)

        # normalize smile_curve by face scale so it is comparable
        smile_curve_n = smile_curve / (nose_chin+1e-6)

        return {
            "MAR": MAR,
            "mouth_w": mouth_w,
            "smile_curve": smile_curve_n,   # >0 corners higher than lip center
            "eye_open": eye_open,
            "brow_gap": brow_gap,
            "brow_raise": brow_raise,       # normalized by face scale
            "nose_upperlip": nose_upperlip, # normalized by face scale
            "nose_chin": nose_chin,
        }

    # ---------- baselines / prototypes ----------
    def set_baseline(self, lm, w, h):
        ft = self._features(lm, w, h)
        self.baseline = dict(ft)  # copy

    def set_proto(self, lm, w, h, label):
        assert label in EMOTIONS
        ft = self._features(lm, w, h)
        self.protos[label] = dict(ft)

    def reset_protos(self):
        self.protos = {}

    # ---------- scoring ----------
    def _gate(self, x, lo, hi):
        # soft gate in [0,1]
        return clamp((x - lo) / max(1e-6, (hi - lo)))

    def _heuristic_scores(self, d):
        # surprise gating: requires BOTH open mouth and wide eyes
        sMAR = self._gate(d["dMAR"], 0.18, 0.35)
        sEye = self._gate(d["dEye"], 0.12, 0.25)
        sBrow = self._gate(d["brow_raise"], 0.05, 0.18)
        surprise = 0.6*sMAR + 0.3*sEye + 0.1*sBrow

        # happy: corners up + wider mouth; penalize if mouth is very open (that leans toward surprise)
        sc = self._gate(d["smile_curve"], 0.10, 0.25)
        mw = self._gate(d["dMouthW"], 0.05, 0.20)
        mar_down = self._gate(-d["dMAR"], 0.02, 0.10)
        happy = (0.7*sc + 0.25*mw + 0.05*mar_down) * (1.0 - 0.5*sMAR)

        # anger: brows closer, eyes narrower, lips pressed
        anger = 0.55*self._gate(-d["dBrowGap"], 0.05, 0.18) + 0.30*self._gate(-d["dEye"], 0.04, 0.12) + 0.15*self._gate(-d["dMAR"], 0.03, 0.12)

        # sad: corners down, droopy eyes, little/no brow raise
        sad = 0.55*self._gate(-d["smile_curve"], 0.08, 0.22) + 0.30*self._gate(-d["dEye"], 0.04, 0.12) + 0.15*self._gate(-d["brow_raise"], -0.02, 0.03)

        # fear: wide eyes + open mouth + raised brows (but less mouth than surprise)
        fear = 0.5*sEye + 0.25*sMAR + 0.25*sBrow

        # disgust: upper lip lift + slight eye narrowing + corners not up
        disg_up = self._gate((self.baseline.get("nose_upperlip", 0.0) - d.get("nose_upperlip_abs", 0.0)), 0.02, 0.10)
        disgust = 0.6*disg_up + 0.25*self._gate(-d["dEye"], 0.02, 0.10) + 0.15*self._gate(-d["smile_curve"], 0.02, 0.12)

        return {"happy":happy, "surprise":surprise, "anger":anger, "sad":sad, "fear":fear, "disgust":disgust}

    def _proto_similarity(self, ft):
        # optional similarity to captured prototypes (per-emotion)
        sims = {k: 0.0 for k in EMOTIONS}
        if not self.protos or self.baseline is None:
            return sims
        # per-emotion feature emphasis
        feats = {
            "happy": ["smile_curve","mouth_w","MAR"],
            "surprise": ["MAR","eye_open","brow_raise"],
            "anger": ["brow_gap","eye_open","MAR"],
            "sad": ["smile_curve","eye_open","brow_raise"],
            "fear": ["eye_open","MAR","brow_raise"],
            "disgust": ["nose_upperlip","eye_open","smile_curve"],
        }
        for lab, proto in self.protos.items():
            if lab not in EMOTIONS:
                continue
            num, den = 0.0, 0.0
            for f in feats[lab]:
                p = proto.get(f, 0.0); x = ft.get(f, 0.0); b = self.baseline.get(f, 1e-6)
                scale = abs(p - b) + 1e-6  # how far the prototype sits from neutral
                num += max(0.0, 1.0 - min(1.0, abs(x - p) / (2.0*scale)))
                den += 1.0
            sims[lab] = num / den if den > 0 else 0.0
        return sims

    # ---------- main update ----------
    def update(self, face_landmarks, w, h):
        if face_landmarks is None:
            return {"emotion": "neutral", "score": 0.0, "drivers": []}
        lm = face_landmarks.landmark

        if self.baseline is None:
            # Auto-capture a neutral-ish baseline; you can press 'b' to recalibrate.
            self.set_baseline(lm, w, h)
            if self._boot_at is None:
                self._boot_at = time.time()
            if time.time() - self._boot_at < 1.0:
                return {"emotion": "neutral", "score": 0.0, "drivers": ["auto-baseline: hold neutral for a sec or press 'b'"]}

        ft = self._features(lm, w, h)
        self.last_ft = dict(ft)
        eps = 1e-6
        d = {
            "dMAR": (ft["MAR"] - self.baseline["MAR"]) / (self.baseline["MAR"] + eps),
            "dMouthW": (ft["mouth_w"] - self.baseline["mouth_w"]) / (self.baseline["mouth_w"] + eps),
            "dEye": (ft["eye_open"] - self.baseline["eye_open"]) / (self.baseline["eye_open"] + eps),
            "dBrowGap": (ft["brow_gap"] - self.baseline["brow_gap"]) / (self.baseline["brow_gap"] + eps),
            "brow_raise": ft["brow_raise"],
            "smile_curve": ft["smile_curve"],
            "nose_upperlip_abs": ft["nose_upperlip"],
        }
        self.last_d = dict(d)

        heur = self._heuristic_scores(d)
        proto = self._proto_similarity(ft)

        combined = {}
        for k in EMOTIONS:
            combined[k] = 0.7*heur.get(k, 0.0) + 0.3*proto.get(k, 0.0)
            self.smooth_scores[k] = ema(self.smooth_scores[k], combined[k], alpha=0.25)
        self.last_scores = dict(combined)

        label = max(self.smooth_scores, key=lambda k: self.smooth_scores[k] or 0.0)
        score = float(self.smooth_scores[label] or 0.0)
        if score < 0.50:
            label = "neutral"; score = 0.0
        self.last_label = label

        # Simple explanation
        drivers = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return {"emotion": label, "score": round(score, 2), "drivers": [f"{k}:{v:.2f}" for k,v in drivers]}

    # ---------- debug ----------
    def debug_values(self):
        """Return last computed features, deltas, and scores for on-screen display."""
        return {
            "features": self.last_ft or {},
            "deltas": self.last_d or {},
            "scores": self.last_scores or {},
            "protos": list(self.protos.keys()),
        }


    # ---------- config io ----------
    def load_config(self, cfg: dict):
        # \"\"\"Load stored baseline and prototypes from a config dict.\"\"\"
        if not isinstance(cfg, dict):
            return
        bl = cfg.get("baseline")
        if isinstance(bl, dict):
            self.baseline = dict(bl)
        prot = cfg.get("protos", {})
        if isinstance(prot, dict):
            # ensure only known emotions are kept
            self.protos = {k: dict(v) for k, v in prot.items() if k in EMOTIONS and isinstance(v, dict)}
    
    def export_config(self) -> dict:
        # \"\"\"Export current baseline and prototypes to a serializable dict.\"\"\"
        return {
            "baseline": self.baseline if isinstance(self.baseline, dict) else None,
            "protos": {k: dict(v) for k, v in (self.protos or {}).items() if k in EMOTIONS},
        }
    