import math
import numpy as np
import cv2
from collections import deque
from utils.common import dist, ema

class BodyLanguageHeuristics:
    """
    Lightweight, explainable rules over pose/hand/face landmarks.
    Press 'b' in the app to set a personal baseline for posture/shrug.
    """
    def __init__(self):
        self.baseline = None
        self.head_x_hist = deque(maxlen=30)
        self.head_y_hist = deque(maxlen=30)
        self.smooth = {k: None for k in [
            "upright","arms_crossed","hand_to_face","nod","shake","shrug"
        ]}

    def set_baseline(self, kp):
        l_sh, r_sh = kp["l_shoulder"], kp["r_shoulder"]
        l_hp, r_hp = kp["l_hip"], kp["r_hip"]
        sh_mid = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2)
        hp_mid = ((l_hp[0]+r_hp[0])//2, (l_hp[1]+r_hp[1])//2)
        shoulder_width = max(1.0, dist(l_sh, r_sh))
        torso_len = max(1.0, dist(sh_mid, hp_mid))
        self.baseline = {
            "shoulder_y": (l_sh[1] + r_sh[1]) / 2.0,
            "torso_len": torso_len,
            "shoulder_width": shoulder_width
        }

    def update(self, kp, draw=False, frame=None):
        labels, confs = {}, {}
        # Head motion (use nose if present)
        nose = kp.get("nose_face", kp.get("nose"))
        if nose:
            self.head_x_hist.append(nose[0])
            self.head_y_hist.append(nose[1])

        req = ["l_shoulder","r_shoulder","l_hip","r_hip"]
        if not all(k in kp for k in req):
            return labels, confs

        l_sh,r_sh,l_hp,r_hp = kp["l_shoulder"],kp["r_shoulder"],kp["l_hip"],kp["r_hip"]
        sh_mid = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2)
        hp_mid = ((l_hp[0]+r_hp[0])//2, (l_hp[1]+r_hp[1])//2)

        shoulder_width = max(1.0, dist(l_sh, r_sh))
        torso_len = max(1.0, dist(sh_mid, hp_mid))

        # 1) Upright vs slouch
        dx = sh_mid[0] - hp_mid[0]
        dy = hp_mid[1] - sh_mid[1]
        torso_angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)+1e-9))
        upright_score = max(0.0, 1.0 - (torso_angle_from_vertical / 25.0))
        if self.baseline:
            upright_score *= min(1.0, (torso_len / (self.baseline["torso_len"]*0.9)))
        self.smooth["upright"] = ema(self.smooth["upright"], upright_score)
        labels["posture"] = "upright" if (self.smooth["upright"] or 0) > 0.55 else "slouch/lean"
        confs["posture_conf"] = round(float(self.smooth["upright"] or 0), 2)

        # 2) Arms crossed
        ac_score = 0.0
        if all(k in kp for k in ["l_wrist","r_wrist","l_elbow","r_elbow"]):
            lw,rw,le,re = kp["l_wrist"],kp["r_wrist"],kp["l_elbow"],kp["r_elbow"]
            chest = sh_mid
            wdist = dist(lw, rw) / shoulder_width
            lw_chest = dist(lw, chest) / shoulder_width
            rw_chest = dist(rw, chest) / shoulder_width
            lw_re = dist(lw, re) / shoulder_width
            rw_le = dist(rw, le) / shoulder_width
            closeness = max(0, 1.2 - wdist)
            chestness = max(0, 1.0 - 0.5*(lw_chest + rw_chest))
            crisscross = max(0, 1.0 - 0.5*(lw_re + rw_le))
            ac_score = np.clip(0.4*closeness + 0.3*chestness + 0.3*crisscross, 0, 1)
        self.smooth["arms_crossed"] = ema(self.smooth["arms_crossed"], ac_score)
        labels["arms"] = "crossed" if (self.smooth["arms_crossed"] or 0) > 0.6 else "open/neutral"
        confs["arms_conf"] = round(float(self.smooth["arms_crossed"] or 0), 2)

        # 3) Hand-to-face
        htf_score = 0.0
        if nose and all(k in kp for k in ["l_wrist","r_wrist"]):
            dmin = min(dist(kp["l_wrist"], nose), dist(kp["r_wrist"], nose)) / shoulder_width
            htf_score = np.clip(1.3 - dmin, 0, 1)
        self.smooth["hand_to_face"] = ema(self.smooth["hand_to_face"], htf_score)
        labels["hands_face"] = "touch/near" if (self.smooth["hand_to_face"] or 0) > 0.6 else "away"
        confs["hands_face_conf"] = round(float(self.smooth["hand_to_face"] or 0), 2)

        # 4) Shoulder shrug
        shrug_score = 0.0
        if self.baseline:
            sh_y = (l_sh[1] + r_sh[1]) / 2.0
            rise = (self.baseline["shoulder_y"] - sh_y) / (self.baseline["torso_len"] + 1e-9)
            shrug_score = np.clip(rise*3.0, 0, 1)
        self.smooth["shrug"] = ema(self.smooth["shrug"], shrug_score)
        labels["shoulders"] = "shrug" if (self.smooth["shrug"] or 0) > 0.55 else "neutral"
        confs["shoulders_conf"] = round(float(self.smooth["shrug"] or 0), 2)

        # 5) Head nod vs shake (oscillation proxy)
        def osc_score(vals):
            if len(vals) < 10: return 0.0
            arr = np.array(vals, dtype=float); arr -= np.mean(arr)
            zc = np.sum(arr[:-1]*arr[1:] < 0)
            amp = np.percentile(np.abs(arr), 90) + 1e-6
            return np.clip((zc/10.0) * (amp/12.0), 0, 1)
        nod_s = osc_score(self.head_y_hist)
        shake_s = osc_score(self.head_x_hist)
        nod_s = max(0.0, nod_s - 0.3*shake_s)
        shake_s = max(0.0, shake_s - 0.3*nod_s)
        self.smooth["nod"] = ema(self.smooth["nod"], nod_s)
        self.smooth["shake"] = ema(self.smooth["shake"], shake_s)
        labels["head"] = "nod" if (self.smooth["nod"] or 0) > 0.55 else ("shake" if (self.smooth["shake"] or 0) > 0.55 else "neutral")
        confs["head_nod_conf"] = round(float(self.smooth["nod"] or 0), 2)
        confs["head_shake_conf"] = round(float(self.smooth["shake"] or 0), 2)

        # Optional overlay
        if draw and frame is not None:
            self.draw_overlay(frame, labels, confs)
        return labels, confs

    def draw_overlay(self, frame, labels, confs):
        h = frame.shape[0]
        panel_x, panel_y = 10, 10
        lines = [
            f"Posture: {labels.get('posture','?')}  ({confs.get('posture_conf',0)})",
            f"Arms: {labels.get('arms','?')}  ({confs.get('arms_conf',0)})",
            f"Hands->Face: {labels.get('hands_face','?')}  ({confs.get('hands_face_conf',0)})",
            f"Head: {labels.get('head','?')}  (nod {confs.get('head_nod_conf',0)}, shake {confs.get('head_shake_conf',0)})",
            f"Shoulders: {labels.get('shoulders','?')}  ({confs.get('shoulders_conf',0)})",
        ]
        (tw, th), _ = cv2.getTextSize("X"*52, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (panel_x-5, panel_y-5), (panel_x+tw+10, panel_y+len(lines)*(th+8)+10), (0,0,0), -1)
        y = panel_y
        for ln in lines:
            cv2.putText(frame, ln, (panel_x, y+th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            y += th+8
        cv2.putText(frame, "Keys: q/Esc quit | b baseline | l log | h helpers", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

class RuleLabeler:
    """Map low-level cues to high-level social states (no ML)."""
    def __init__(self):
        self.smooth_score = None
        self.smooth_state = None

    def infer(self, labels, confs):
        upright = float(confs.get("posture_conf", 0.0)); slouch = 1.0 - upright
        arms_crossed = float(confs.get("arms_conf", 0.0)); arms_open = 1.0 - arms_crossed
        hands_face = float(confs.get("hands_face_conf", 0.0)); no_self_touch = 1.0 - hands_face
        shrug = float(confs.get("shoulders_conf", 0.0))
        nod = float(confs.get("head_nod_conf", 0.0))
        shake = float(confs.get("head_shake_conf", 0.0))

        scores = {
            "confident": 0.35*upright + 0.25*arms_open + 0.15*no_self_touch + 0.15*(1.0 - shrug) + 0.10*nod,
            "attentive": 0.35*upright + 0.25*nod + 0.20*arms_open + 0.20*no_self_touch,
            "closed_off": 0.40*arms_crossed + 0.30*slouch + 0.15*(1.0 - nod) + 0.15*(1.0 - no_self_touch),
            "uncertain": 0.40*shrug + 0.30*hands_face + 0.20*shake + 0.10*slouch,
            "stressed": 0.40*hands_face + 0.25*shake + 0.20*shrug + 0.15*slouch,
            "disengaged": 0.45*slouch + 0.25*(1.0 - nod) + 0.15*arms_crossed + 0.15*(1.0 - no_self_touch),
        }
        best_label = max(scores, key=scores.get)
        best_score = np.clip(scores[best_label], 0, 1)
        alpha = 0.25
        self.smooth_score = alpha*best_score + (1-alpha)*(self.smooth_score if self.smooth_score is not None else best_score)
        self.smooth_state = best_label if (self.smooth_state is None or best_label == self.smooth_state) else (
            best_label if best_score > (self.smooth_score or 0) else self.smooth_state
        )
        final_label = self.smooth_state if (self.smooth_score or 0) >= 0.55 else "neutral"
        final_score = round(float(self.smooth_score or 0), 2)
        return {"state": final_label, "score": final_score}
