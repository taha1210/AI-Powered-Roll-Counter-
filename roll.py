import cv2
import math
import numpy as np
from ultralytics import YOLO
from sort import Sort   # ✅ tumne already bana liya hai sort.py

# ---- Model load ----
model = YOLO(r"C:\Users\taha.amjad\Downloads\rollbest.pt")

# ---- RTSP Camera URL ----
IP_CAMERA_URL = "rtsp://admin:admin123@192.168.28.156:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(IP_CAMERA_URL)

# ✅ Fix #2: Buffer clear karo (no frame lag accumulation)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("⚠️ Camera stream open nahi ho rahi, IP/credentials check karo")
    exit()

# ---- Counter ----
total_counter = 0
roll_states = {}          # {track_id: "forward"/"reverse"}
prev_positions = {}       # {track_id: (cx,cy)}

# ---- Virtual Line ----
line_start = (2541, 676)
line_end   = (2334, 643)

def point_side_of_line(p, a, b):
    """Return >0 if left, <0 if right, 0 if on the line"""
    return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

# ---- SORT Tracker ----
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.1)

frame_id = 0   # ✅ Fix #1: frame skip variable

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read fail, stream toot gayi shayad")
        break

    frame_id += 1
    # ✅ Fix #1: Har dusra frame process karo (FPS double)
    if frame_id % 2 != 0:
        continue

    # ✅ Fix #3: Smaller input size for faster inference
    results = model.predict(frame, imgsz=480, conf=0.35, verbose=False)

    # YOLO detections
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections, dtype=float)

    # ✅ Always keep 2D shape (N, 5)
    if detections.ndim == 1:
        if detections.size == 0:
            detections = np.empty((0, 5))
        else:
            detections = detections.reshape(1, 5)

    # ✅ Safe SORT update
    if len(detections) == 0:
        tracks = []
    else:
        tracks = tracker.update(detections)

    for x1, y1, x2, y2, tid in tracks:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        # Agar pehle position save hai
        if tid in prev_positions:
            prev = prev_positions[tid]

            prev_side = point_side_of_line(prev, line_start, line_end)
            curr_side = point_side_of_line((cx, cy), line_start, line_end)

            if prev_side * curr_side < 0:  # ✅ Line crossed
                if curr_side < prev_side:
                    #if prev_side:   # ✅ Forward
                    if roll_states.get(tid) != "forward":
                        total_counter += 1
                        roll_states[tid] = "forward"
                        print(f"➡️ Roll {int(tid)} Forward | Total:{total_counter}")
                else:  # ✅ Reverse
                    if roll_states.get(tid) != "reverse":
                        total_counter -= 1
                        roll_states[tid] = "reverse"
                        print(f"⬅️ Roll {int(tid)} Reverse | Total:{total_counter}")

        prev_positions[tid] = (cx, cy)

        # Draw box + ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
        cv2.putText(frame, f"ID {int(tid)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Draw line
    cv2.line(frame, line_start, line_end, (0,0,255), 4)

    # Draw total counter
    cv2.putText(frame, f"Total Rolls: {total_counter}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # ✅ Fix #4: Resizable live feed
    cv2.namedWindow("Roll Counter - Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Roll Counter - Live Feed", 1280, 720)
    cv2.imshow("Roll Counter - Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Live stream processing complete.")
