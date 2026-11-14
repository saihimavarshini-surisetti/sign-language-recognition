"""Real-time sign language recognition.

Controls:
    q  quit        c  clear sentence
    b  backspace   s  speak sentence (text-to-speech)
A letter is appended automatically after being held steadily (~1 second).
"""
import argparse, time
from collections import deque, Counter
import cv2, joblib, numpy as np
from utils import create_hands, extract_features, draw_hand


def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text); engine.runAndWait()
    except Exception as e:
        print("TTS unavailable:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="models/sign_model.pkl")
    ap.add_argument("--conf", type=float, default=0.6, help="min confidence")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to confirm a sign")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, classes = bundle["model"], bundle["classes"]
    hands = create_hands()
    cap = cv2.VideoCapture(args.camera)

    history = deque(maxlen=10)       # smoothing window
    sentence = ""
    stable_label, stable_since, added = None, 0, False
    prev_t = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats, res = extract_features(frame, hands)
        frame = draw_hand(frame, res)

        label, conf = None, 0.0
        if feats is not None:
            proba = model.predict_proba([feats])[0]
            idx = int(np.argmax(proba))
            if proba[idx] >= args.conf:
                history.append(classes[idx]); conf = float(proba[idx])
                label = Counter(history).most_common(1)[0][0]
        else:
            history.clear()

        now = time.time()
        if label and label == stable_label:
            if not added and now - stable_since >= args.hold:
                sentence += label if len(label) == 1 else f" {label} "
                added = True
        else:
            stable_label, stable_since, added = label, now, False

        fps = 1 / max(now - prev_t, 1e-6); prev_t = now
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(frame, f"Sign: {label or '-'} ({conf:.0%})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {sentence}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Sign Language Recognition", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        elif k == ord("c"): sentence = ""
        elif k == ord("b"): sentence = sentence[:-1]
        elif k == ord("s") and sentence.strip(): speak(sentence)

    cap.release(); cv2.destroyAllWindows(); hands.close()


if __name__ == "__main__":
    main()
