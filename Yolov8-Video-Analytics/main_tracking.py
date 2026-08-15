from ultralytics import YOLO
import cv2
import pandas as pd
import time
from deepface import DeepFace

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.5 
ENABLE_AGE_GENDER = True    
VIDEO_PATH = "inputs/Berghouse Leopard Jog.mp4"

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Cannot open video file {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup output video writer
output_video = 'outputs/output_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Initialize tracking variables
csv_data = []
detect_persons = True
frame_count = 0
age_gender_cache = {}

#Analyze age and gender using DeepFace
def analyze_age_gender(face_roi):
    try:
        result = DeepFace.analyze(face_roi, actions=['age', 'gender'], 
                                 enforce_detection=False, silent=True)
        
        if isinstance(result, list):
            result = result[0]
        
        age = int(result['age'])
        gender = result['dominant_gender'].capitalize()
        return gender, age
    except Exception:
        return "Unknown", "?"

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw, y), color, -1)
    cv2.putText(frame, text, (x, y - 4), font, font_scale, (255, 255, 255), thickness)

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    start_time = time.time()
    
    results = model.track(frame, persist=True, verbose=False, tracker="botsort.yaml", conf=CONFIDENCE_THRESHOLD,iou=0.5)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
            label = model.names[cls_id]
            
            # Filter for persons only
            if detect_persons and label != "person":
                continue
            
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            
            gender, age = "-", "-"
            
            # Perform age/gender detection for persons
            if label == "person" and ENABLE_AGE_GENDER:
                if track_id in age_gender_cache:
                    gender, age = age_gender_cache[track_id]
                else:
                    # Extract face region from upper portion of bounding box
                    face_y1 = y1
                    face_y2 = y1 + int(height * 0.4)
                    face_roi = frame[face_y1:face_y2, x1:x2]
                    
                    if face_roi.size > 0 and width > 30 and height > 50:
                        gender, age = analyze_age_gender(face_roi)
                        age_gender_cache[track_id] = (gender, age)
            
            if label == "person":
                display_label = f"ID:{track_id} | {gender} | {age}"
                color = (0, 255, 255)
            else:
                display_label = f"{label} ({conf:.2f})"
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, display_label, x1, y1, color)

            timestamp = round(frame_count / fps, 2)
            csv_data.append([timestamp, frame_count, track_id, label, 
                           round(conf, 3), gender, age, x1, y1, width, height])
    
    processing_time = time.time() - start_time
    current_fps = 1.0 / processing_time if processing_time > 0 else 0
    cv2.putText(frame, f"FPS: {current_fps:.1f} | Frame: {frame_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

df = pd.DataFrame(csv_data, columns=["Timestamp", "Frame", "TrackID", "Class", 
                                      "Confidence", "Gender", "Age", "X", "Y", "Width", "Height"])
df.to_csv("outputs/detections.csv", index=False)

print(f"\nProcessing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Output video saved: {output_video}")
print(f"Detections saved: detections.csv")
print(f"Total detections: {len(csv_data)}")