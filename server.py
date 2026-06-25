import socket
import pickle
import struct
import cv2
import numpy as np
import threading
import time
import struct

from ultralytics import FastSAM
from config import FASTSAM_MODEL, IMAGE_SIZE, DEVICE, PORT, THRESHOLD_UNKNOWN, MIN_MASK_PIXELS
from clip_classifier import ClipClassifier
from vlm import generate_labels
from utils import mask_crop, draw_label, send_frame



# MODELS
seg = FastSAM(FASTSAM_MODEL)
clf = ClipClassifier()
print("Prêt")


# STATE VLM
vlm_status = "idle"
vlm_labels = []
freeze_until = 0
frame_ref = [None]


# KEYBOARD THREAD
def keyboard_thread():
    global vlm_status, vlm_labels, freeze_until

    print("\nTap 'l' + Enter to launch VLM")

    while True:
        cmd = input().strip().lower()

        if cmd == "l":
            frame = frame_ref[0]

            if frame is None:
                print("❌ no frame available")
                continue

            print("🧠 VLM launching...")
            vlm_status = "running"
            freeze_until = time.time() + 2

            try:
                labels = generate_labels(frame)
                vlm_labels = labels
                clf.update_labels(labels)

                print("✅ Labels:", labels)
                vlm_status = "done"

            except Exception as e:
                print("❌ Erreur VLM:", e)
                vlm_status = "error"


threading.Thread(target=keyboard_thread, daemon=True).start()



# SOCKET
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", PORT))
server_socket.listen(1)

conn, addr = server_socket.accept()
print("Client connected:", addr)

data = b""
payload_size = struct.calcsize("Q")


def recv_exact(n):
    global data
    while len(data) < n:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk
    out = data[:n]
    data = data[n:]
    return out


def build_payload(frame, send_time):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not ok:
        return {"frame": None, "send_time": send_time}

    return {
        "frame": buf,
        "send_time": send_time
    }


# LOOP
try:
    while True:

        try:
            header = recv_exact(payload_size)
            msg_size = struct.unpack("Q", header)[0]

            if msg_size > 20_000_000:
                data = b""
                continue

            raw = recv_exact(msg_size)
            frame, send_time = pickle.loads(raw)

            frame_ref[0] = frame.copy()

        except Exception as e:
            print("Error receiving:", e)
            data = b""
            continue



        # FASTSAM
        results = seg.track(
            frame,
            imgsz=640,          # Input image size for inference (resizes frame to 640x640)
            save=False,         # Do not save output images/videos to disk
            show=False,         # Do not display visualization window
            conf=0.8,           # Confidence threshold for detections (only keep predictions >= 0.8)
            persist=True,       # Persist tracks between frames (important for video tracking)
            tracker="bytetrack.yaml",  # Tracking algorithm configuration (ByteTrack)
            iou=0.5,            # IoU threshold for matching detections across frames
            max_det=20,         # Maximum number of detections per frame
            half=True,          # Use FP16 precision (faster inference on GPU, less memory)
            device=DEVICE       # Device to run inference on (e.g., 'cpu', 'cuda:0')
        )

        annotated = results[0].plot()

        # VLM OVERLAY
        cv2.putText(
            annotated,
            f"VLM: {vlm_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        y = 60
        for l in vlm_labels:
            cv2.putText(
                annotated,
                str(l),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y += 25



        # OBJECT DETECTION + CLIP ALL OBJECTS
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            center = np.array([w // 2, h // 2])

            objects = []

            # collect all objects 
            for i in range(masks.shape[0]):
                mask = masks[i].astype(bool)
                ys, xs = np.where(mask)

                if len(xs) < MIN_MASK_PIXELS:
                    continue

                cx, cy = int(xs.mean()), int(ys.mean())
                crop = mask_crop(frame, mask, xs, ys)

                label, score = clf.predict(crop)

                if score < THRESHOLD_UNKNOWN:
                    label = "unknown"

                dist = np.linalg.norm(center - np.array([cx, cy]))

                objects.append({
                    "label": label,
                    "score": score,
                    "center": (cx, cy),
                    "dist": dist
                })

            # find most central 
            best_obj = None
            if objects:
                best_obj = min(objects, key=lambda x: x["dist"])

            # draw all
            for obj in objects:
                cx, cy = obj["center"]
                label = obj["label"]
                score = obj["score"]

                is_best = (obj == best_obj)

                color = (0, 255, 0) if is_best else (180, 180, 180)

                text = f"> {label} {score:.2f}" if is_best else f"{label} {score:.2f}"

                draw_label(annotated, text, (cx, cy), color=color)

        # FREEZE LOGIC
        if time.time() < freeze_until:
            send_frame(conn, build_payload(annotated, send_time))
            continue


        # SEND
        try:
            send_frame(conn, build_payload(annotated, send_time))
        except Exception as e:
            print("Error sending:", e)
            break


except KeyboardInterrupt:
    print("Stop requested")

finally:
    conn.close()
    server_socket.close()
    print("Server closed")