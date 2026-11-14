"""Collect training samples from your webcam.

Usage:
    python collect_data.py                 # A-Z
    python collect_data.py --labels HELLO,YES,NO,THANKS

Controls:
    Press a key matching a label's first character (e.g. 'a') to select it.
    For custom word labels use number keys 1-9 (order given in --labels).
    SPACE  -> hold to record samples continuously for the selected label
    q      -> quit and save
"""
import argparse, os, string
import cv2, numpy as np, pandas as pd
from utils import create_hands, extract_features, draw_hand

CSV_PATH = os.path.join("data", "landmarks.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=",".join(string.ascii_uppercase))
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    custom = any(len(l) > 1 for l in labels)

    rows = []
    counts = {l: 0 for l in labels}
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        for l, c in old["label"].value_counts().items():
            counts[l] = counts.get(l, 0) + int(c)

    cap = cv2.VideoCapture(args.camera)
    hands = create_hands(max_hands=1)
    current, recording = None, False
    print("Labels:", labels)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats, res = extract_features(frame, hands)
        frame = draw_hand(frame, res)

        if recording and current and feats is not None:
            rows.append([current] + feats.tolist())
            counts[current] += 1

        cv2.putText(frame, f"Label: {current}  Samples: {counts.get(current, 0)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "REC" if recording else "SPACE=record  q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recording = not recording
        elif key != 255:
            ch = chr(key)
            if custom and ch.isdigit() and 1 <= int(ch) <= len(labels):
                current = labels[int(ch) - 1]; recording = False
            elif not custom and ch.upper() in labels:
                current = ch.upper(); recording = False

    cap.release(); cv2.destroyAllWindows(); hands.close()

    if rows:
        cols = ["label"] + [f"f{i}" for i in range(63)]
        df = pd.DataFrame(rows, columns=cols)
        if os.path.exists(CSV_PATH):
            df = pd.concat([pd.read_csv(CSV_PATH), df], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved. Total samples: {len(df)}")
        print(df['label'].value_counts().to_string())
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()
