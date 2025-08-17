
import cv2
import json
import time
import mediapipe as mp
from pathlib import Path
from detectors.emotion import EmotionHeuristics, EMOTIONS

WIN = "Calibration"

STEPS = ["neutral"] + EMOTIONS

PROMPTS = {
    "neutral": "Relax face, natural expression.",
    "happy": "Smile naturally (corners up), mouth lightly open or closed.",
    "surprise": "Eyes wide, jaw dropped a bit.",
    "anger": "Brows inward/down, lips pressed.",
    "sad": "Corners down, droopy eyes.",
    "fear": "Eyes wide, brows raised, mouth slightly open.",
    "disgust": "Wrinkle nose, raise upper lip.",
}

def draw_center_text(img, lines, y0=40):
    y = y0
    for ln in lines:
        cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        y += 36

def main():
    out_path = Path("emotion_config.json")
    emo = EmotionHeuristics()
    hol = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam"); return

    captured = {}
    idx = 0

    with hol.Holistic(model_complexity=1, refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        while idx < len(STEPS):
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            hgt, wdt = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = h.process(rgb)
            rgb.flags.writeable = True

            if res.face_landmarks:
                mp_draw.draw_landmarks(
                    frame, res.face_landmarks, hol.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            step = STEPS[idx]
            prompt = PROMPTS.get(step, step)

            # UI overlay
            cv2.rectangle(frame, (0,0), (wdt, 115), (32,32,32), -1)
            draw_center_text(frame, [f"Calibration {idx+1}/{len(STEPS)}: {step.upper()}", prompt, "Press SPACE to capture, or ESC to quit"], y0=32)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == 32:  # space
                if res.face_landmarks:
                    if step == "neutral":
                        emo.set_baseline(res.face_landmarks.landmark, wdt, hgt)
                        captured["baseline"] = emo.baseline
                        idx += 1
                    else:
                        emo.set_proto(res.face_landmarks.landmark, wdt, hgt, step)
                        captured.setdefault("protos", {})[step] = emo.protos[step]
                        idx += 1

        # Save if we have at least neutral + one emotion
        if "baseline" in captured and "protos" in captured and len(captured["protos"])>0:
            out_path.write_text(json.dumps(captured, indent=2))
            print(f"Saved calibration to {out_path.resolve()}")
        else:
            print("Calibration incomplete; nothing saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
