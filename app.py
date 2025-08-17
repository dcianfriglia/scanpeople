
import cv2
import csv
import time
import numpy as np
from pathlib import Path

import mediapipe as mp
from utils.common import to_px, FPSCounter
from detectors.body_language import BodyLanguageHeuristics, RuleLabeler
from detectors.emotion import EmotionHeuristics, EMOTIONS

WIN = "Body+Emotion (demo)"

# --- UI helpers ---
def draw_text_lines(img, lines, x, y, line_h, color=(0,255,255)):
    """Draw a list of strings onto img starting at (x,y). Returns next y."""
    for ln in lines:
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += line_h
    return y

def make_sidebar(w, h, labels, confs, emo_readout, rule_state=None, show_values=False, emo=None):
    """
    Build a right-hand panel of width ~320px containing all text readouts.
    - labels/confs from body language
    - emotion summary
    - optional raw values ('v' toggle)
    """
    side_w = 320
    panel = np.zeros((h, side_w, 3), dtype=np.uint8)

    # Header
    cv2.rectangle(panel, (0,0), (side_w, 36), (32,32,32), -1)
    cv2.putText(panel, "Readout", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    y = 52
    lh = 20

    # Emotion block
    cv2.putText(panel, "Emotion", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    y += lh
    cv2.putText(panel, f"{emo_readout['emotion']} ({emo_readout['score']:.2f})", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    y += lh
    if emo_readout.get("drivers"):
        for d in emo_readout["drivers"]:
            cv2.putText(panel, f"- {d}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
            y += lh

    # Rule-based social state (if provided)
    if rule_state:
        y += 8
        cv2.putText(panel, "Social state", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        y += lh
        cv2.putText(panel, f"{rule_state.get('state','neutral')} ({rule_state.get('score',0):.2f})", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        y += lh

    # Body language block
    y += 8
    cv2.putText(panel, "Body", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    y += lh
    items = [
        ("Posture", labels.get("posture","?"), confs.get("posture_conf",0)),
        ("Arms", labels.get("arms","?"), confs.get("arms_conf",0)),
        ("Hands->Face", labels.get("hands_face","?"), confs.get("hands_face_conf",0)),
        ("Head", labels.get("head","?"), max(confs.get("head_nod_conf",0), confs.get("head_shake_conf",0))),
        ("Shoulders", labels.get("shoulders","?"), confs.get("shoulders_conf",0)),
    ]
    for name, lab, sc in items:
        cv2.putText(panel, f"{name}: {lab} ({sc:.2f})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        y += lh

    # Optional raw values
    if show_values and emo is not None:
        y += 8
        cv2.putText(panel, "Debug values", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        y += lh
        info = emo.debug_values()
        ft = info.get("features", {}); d = info.get("deltas", {}); sc = info.get("scores", {})
        # features
        for k in ["MAR","mouth_w","smile_curve","eye_open","brow_gap","brow_raise","nose_upperlip"]:
            v = ft.get(k, None)
            cv2.putText(panel, f"{k}: {v:.3f}" if v is not None else f"{k}: -", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)
            y += 16
        # deltas
        for k in ["dMAR","dMouthW","dEye","dBrowGap","smile_curve"]:
            v = d.get(k, None)
            cv2.putText(panel, f"{k}: {v:.3f}" if v is not None else f"{k}: -", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)
            y += 16
        # scores
        for k in EMOTIONS:
            v = sc.get(k, 0.0)
            cv2.putText(panel, f"{k}: {v:.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)
            y += 16

    # Footer tips
    y = max(y+8, h-40)
    tip = "Keys: q/Esc | b baseline | 1..6 protos | v values | h helpers | l log | r reset"
    cv2.putText(panel, tip, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)

    return panel

def cleanup(cap=None, log_file=None):
    try:
        if log_file and not log_file.closed:
            log_file.flush(); log_file.close()
    except Exception:
        pass
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1); time.sleep(0.01)
    except Exception:
        pass

def extract_keypoints(results, w, h):
    kp = {}
    if results.pose_landmarks:
        idx = mp.solutions.pose.PoseLandmark
        pose = results.pose_landmarks.landmark
        for name, lid in {
            "nose": idx.NOSE,
            "l_shoulder": idx.LEFT_SHOULDER, "r_shoulder": idx.RIGHT_SHOULDER,
            "l_elbow": idx.LEFT_ELBOW, "r_elbow": idx.RIGHT_ELBOW,
            "l_wrist": idx.LEFT_WRIST, "r_wrist": idx.RIGHT_WRIST,
            "l_hip": idx.LEFT_HIP, "r_hip": idx.RIGHT_HIP
        }.items():
            kp[name] = to_px(pose[lid], w, h)
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        kp["nose_face"] = to_px(face[1], w, h)
    return kp


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam"); return

    bl = BodyLanguageHeuristics()
    rule = RuleLabeler()
    emo = EmotionHeuristics()

    draw_helpers = True
    show_values = False  # default off in sidebar
    logging = False
    out_dir = Path("../captures"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_writer = None
    log_file = None
    fps = FPSCounter()

    try:
        with holistic.Holistic(
            model_complexity=1,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hol:
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hol.process(rgb)
                rgb.flags.writeable = True

                # Draw helpers
                if draw_helpers:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                        )
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                        )

                kp = extract_keypoints(results, w, h)
                # Do NOT draw overlays on the video frame; we will use a sidebar.
                labels, confs = bl.update(kp, draw=False, frame=None)

                # Emotion module
                face_lms = results.face_landmarks
                hl = {"emotion": "neutral", "score": 0.0, "drivers": []}
                if face_lms:
                    hl = emo.update(face_lms, w, h)

                # High-level social state from rules
                rule_state = rule.infer(labels, confs) if labels else {"state":"neutral", "score":0.0}

                # FPS (draw on video only)
                cv2.putText(frame, f"FPS: {fps.tick():.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                # Compose sidebar and show
                sidebar = make_sidebar(w, h, labels, confs, hl, rule_state, show_values, emo)
                composite = cv2.hconcat([frame, sidebar])

                cv2.imshow(WIN, composite)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                try:
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

                # Hotkeys
                if key == ord('b'):
                    if face_lms:
                        emo.set_baseline(face_lms.landmark, w, h)
                    if kp:
                        bl.set_baseline(kp)
                elif key == ord('v'):
                    show_values = not show_values
                elif key == ord('h'):
                    draw_helpers = not draw_helpers
                elif key == ord('r'):
                    emo.reset_protos()
                elif key == ord('l'):
                    if not logging:
                        logging = True
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        log_file = open(out_dir / f"log_{ts}.csv", "w", newline="", encoding="utf-8")
                        fieldnames = [
                            "time","emotion","emotion_score","posture","arms","hands_face","head","shoulders",
                            "posture_conf","arms_conf","hands_face_conf","head_nod_conf","head_shake_conf","shoulders_conf"
                        ]
                        csv_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
                        csv_writer.writeheader()
                    else:
                        logging = False
                        if log_file:
                            log_file.flush(); log_file.close(); log_file = None
                        csv_writer = None
                elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
                    if face_lms:
                        mapping = {
                            ord('1'): 'happy', ord('2'): 'surprise', ord('3'): 'anger',
                            ord('4'): 'sad',   ord('5'): 'fear',     ord('6'): 'disgust',
                        }
                        emo.set_proto(face_lms.landmark, w, h, mapping[key])

                if logging and csv_writer:
                    csv_writer.writerow({
                        "time": time.time(),
                        "emotion": hl.get("emotion",""),
                        "emotion_score": hl.get("score",0),
                        "posture": labels.get("posture",""),
                        "arms": labels.get("arms",""),
                        "hands_face": labels.get("hands_face",""),
                        "head": labels.get("head",""),
                        "shoulders": labels.get("shoulders",""),
                        "posture_conf": confs.get("posture_conf",0),
                        "arms_conf": confs.get("arms_conf",0),
                        "hands_face_conf": confs.get("hands_face_conf",0),
                        "head_nod_conf": confs.get("head_nod_conf",0),
                        "head_shake_conf": confs.get("head_shake_conf",0),
                        "shoulders_conf": confs.get("shoulders_conf",0),
                    })

    finally:
        cleanup(cap=cap, log_file=log_file)

if __name__ == "__main__":
    main()
