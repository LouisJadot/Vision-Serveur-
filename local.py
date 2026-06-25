import socket
import cv2
import pickle
import struct
import time
import threading

SERVER_IP  = "127.0.0.1"
PORT       = 10003
CAMERA_IDX = 2

# SOCKET
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))
print(f"Connecté au serveur {SERVER_IP}:{PORT}")

# CAMERA
cap = cv2.VideoCapture(CAMERA_IDX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Impossible to open the camera")
    exit(1)

payload_size = struct.calcsize("Q")
MAX_MSG_SIZE = 20_000_000

lock = threading.Lock()
last_frame = None
last_latency = 0.0
running = True


def annotate(frame, latency, has_server_frame):
    if has_server_frame:
        cv2.putText(frame, f"{latency:.0f} ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Waiting for server...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


# THREAD RECEPTION
def recv_thread():
    global last_frame, last_latency, running

    data = b""

    def recv_exact(n):
        nonlocal data
        while len(data) < n:
            chunk = client_socket.recv(65536)
            if not chunk:
                raise ConnectionError("Server disconnected")
            data += chunk
        out = data[:n]
        data = data[n:]
        return out

    while running:
        try:
            header = recv_exact(payload_size)
            msg_size = struct.unpack("Q", header)[0]

            if msg_size > MAX_MSG_SIZE:
                data = b""
                continue

            raw = recv_exact(msg_size)
            payload = pickle.loads(raw)

            if isinstance(payload, dict):
                buf = payload["frame"]
                send_time = payload["send_time"]
            else:
                buf, send_time = payload

            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            latency = (time.time() - send_time) * 1000

            with lock:
                last_frame = frame
                last_latency = latency

        except Exception as e:
            print("Error receiving:", e)
            running = False
            break


threading.Thread(target=recv_thread, daemon=True).start()


# LOOP PRINCIPALE
frame_interval = 1.0 / 15
last_send = 0

try:
    while running:
        ret, raw_frame = cap.read()
        if not ret:
            continue

        now = time.time()

        if now - last_send >= frame_interval:
            send_time = time.time()
            message = pickle.dumps((raw_frame, send_time))
            packet = struct.pack("Q", len(message)) + message

            try:
                client_socket.sendall(packet)
                last_send = send_time
            except Exception as e:
                print("Error sending:", e)
                break

        with lock:
            if last_frame is not None:
                display = last_frame.copy()
                latency = last_latency
                has = True
            else:
                display = raw_frame.copy()
                latency = 0
                has = False

        annotate(display, latency, has)

        cv2.imshow("Client", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    running = False
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
    print("Client closed")