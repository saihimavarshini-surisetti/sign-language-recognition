"""Shared helpers: hand landmark extraction and feature normalisation."""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def create_hands(static=False, max_hands=1, det_conf=0.6, track_conf=0.6):
    return mp_hands.Hands(
        static_image_mode=static,
        max_num_hands=max_hands,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )


def normalize_landmarks(hand_landmarks):
    """Convert 21 landmarks into a 63-d vector that is invariant to
    hand position and size (translate to wrist, scale by hand span)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts -= pts[0]                                   # wrist = origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def extract_features(frame_bgr, hands):
    """Returns (features or None, mediapipe results)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        return normalize_landmarks(results.multi_hand_landmarks[0]), results
    return None, results


def draw_hand(frame, results):
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())
    return frame
