from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os
import time

cv2.setNumThreads(1)

# Load YOLOv8n model (lightweight, good for Pi)
model = YOLO("yolov8n.pt")

# Load Caffe models (optional: remove if you don't want age/gender detection)
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Use USB camera from drone (adjust index if needed)
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if 0 doesn't work

# Optional: set resolution to reduce load on Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

video_name = 'drone_feed_capture'
csv_data = []
frame_skip = 3  # Reduce load
frame_count = 0
detect_objects = True  # Always ON
detect_gender = False  # Toggle with 'f'
detect_age = False     # Toggle with 'a'
detect_persons = False  # Toggle with 'p'
paused = False
person_id_counter = 1
person_registry = []
age_gender_cache = {}
resize_dim = (416, 416)

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw, y), color, -1)
    cv2.putText(frame, text, (x, y - 2), font, font_scale, (0, 0, 0), thickness)

def draw_status_panel(frame, detect_objects, detect_persons, detect_gender, detect_age, fps):
    lines = [
        f"FPS: {fps:.2f}",
        f"[O] Object Only: {'ON' if detect_objects else 'OFF'}",
        f"[P] Person Only: {'ON' if detect_persons else 'OFF'}",
        f"[F] Gender: {'ON' if detect_gender else 'OFF'}",
        f"[A] Age: {'ON' if detect_age else 'OFF'}",
        "[Space] Pause  [Q] Quit"
    ]
    x, y0 = 10, 30
    for i, line in enumerate(lines):
        y = y0 + i * 20
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 255), 1, cv2.LINE_AA)

cv2.namedWindow("Drone YOLOv8 Detection", cv2.WINDOW_NORMAL)
log_file = open(f"{video_name}_log.txt", "w")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (frame_skip + 1) != 0:
            continue

        start_time = time.time()
        original_frame = frame.copy()
        frame_resized = cv2.resize(frame, resize_dim)
        results = model(frame_resized, verbose=False)[0]
        detections = results.boxes
        names = model.names

        h_ratio = original_frame.shape[0] / resize_dim[1]
        w_ratio = original_frame.shape[1] / resize_dim[0]

        for i in range(len(detections.cls)):
            label = names[int(detections.cls[i])]
            confidence = float(detections.conf[i])
            if detect_objects and label != "person":
                continue
            if detect_persons and label != "person":
                continue

            box = detections.xyxy[i].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin * w_ratio)
            xmax = int(xmax * w_ratio)
            ymin = int(ymin * h_ratio)
            ymax = int(ymax * h_ratio)
            width, height = xmax - xmin, ymax - ymin
            time_in_sec = round(time.time(), 2)

            bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            matched = False
            person_label = label

            if label == 'person':
                for pid, center in person_registry:
                    if abs(center[0] - bbox_center[0]) < 50 and abs(center[1] - bbox_center[1]) < 50:
                        person_label = f'person{pid}'
                        matched = True
                        break
                if not matched:
                    person_label = f'person{person_id_counter}'
                    person_registry.append((person_id_counter, bbox_center))
                    person_id_counter += 1

                gender, age = "-", "-"
                roi = frame[ymin:ymax, xmin:xmax]
                if roi.size > 0:
                    if person_label in age_gender_cache:
                        gender, age = age_gender_cache[person_label]
                    else:
                        try:
                            blob = cv2.dnn.blobFromImage(cv2.resize(roi, (227, 227)), 1.0, (227, 227),
                                                         (78.4263377603, 87.7689143744, 114.895847746),
                                                         swapRB=False)
                            if detect_gender:
                                gender_net.setInput(blob)
                                gender_preds = gender_net.forward()
                                gender = gender_list[gender_preds[0].argmax()]
                            if detect_age:
                                age_net.setInput(blob)
                                age_preds = age_net.forward()
                                age = age_list[age_preds[0].argmax()]
                            age_gender_cache[person_label] = (gender, age)
                        except:
                            gender, age = "?", "?"
            else:
                gender = age = "-"

            label_display = f"{person_label if label == 'person' else label}"
            if detect_gender and gender != "-":
                label_display += f" | G: {gender}"
            if detect_age and age != "-":
                label_display += f" | A: {age}"

            color = (255, 200, 0) if label == 'person' else (0, 255, 0)
            cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), color, 2)
            draw_label(original_frame, label_display, xmin, ymin, color)

            if label == 'person':
                csv_data.append([time_in_sec, person_label, round(confidence, 2), gender, age, xmin, ymin, width, height])

            log_file.write(f"[{time_in_sec}s] {label_display} @ ({xmin},{ymin},{width},{height})\n")

        fps = 1.0 / (time.time() - start_time)
        draw_status_panel(original_frame, detect_objects, detect_persons, detect_gender, detect_age, fps)
        cv2.imshow("Drone YOLOv8 Detection", original_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('o'):
        detect_objects = not detect_objects
    elif key == ord('p'):
        detect_persons = not detect_persons
    elif key == ord('f'):
        detect_gender = not detect_gender
    elif key == ord('a'):
        detect_age = not detect_age
    elif key == ord('b'):
        detect_objects = detect_persons = detect_gender = detect_age = True

cap.release()
cv2.destroyAllWindows()
log_file.close()

df = pd.DataFrame(csv_data, columns=["Time", "Class", "Confidence", "Gender", "Age", "X", "Y", "Width", "Height"])
df.to_csv(f"{video_name}_detections.csv", index=False)
print(f"Detections saved to {video_name}_detections.csv")
