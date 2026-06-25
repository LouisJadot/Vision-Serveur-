import cv2
import numpy as np
import pickle
import struct

from config import CLIP_PADDING

# MASK CROP 
def mask_crop(frame, mask, xs, ys, padding=CLIP_PADDING):

    if frame is None or mask is None:
        return None

    h, w = frame.shape[:2]

    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, w - 1)

    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2 + 1, x1:x2 + 1].copy()

    # ssecurity to avoid empty crops
    if crop.size == 0:
        return None

    return crop

# DRAW LABEL
def draw_label(img, text, pos, color=(0, 255, 0)):

    if img is None:
        return

    x, y = pos

    (w, h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    # fond
    cv2.rectangle(
        img,
        (x, y - h - 6),
        (x + w, y + 6),
        (0, 0, 0),
        -1
    )

    # texte
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )



# SEND FRAME SOCKET
def send_frame(conn, payload: dict):

    try:
        data = pickle.dumps(payload)
        conn.sendall(struct.pack("Q", len(data)) + data)

    except Exception as e:
        print(f"[send_frame] Error: {e}")