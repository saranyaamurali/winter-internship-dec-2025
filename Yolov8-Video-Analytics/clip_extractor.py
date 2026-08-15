import cv2
import pandas as pd
import numpy as np
import os
import time
from ultralytics import YOLO


VIDEO_PATH = "inputs/city.mp4"       
MODEL_PATH = "models/yolov8m.pt"           
OUTPUT_TRACKED_VIDEO = "outputs/tracked_output_city.mp4"    
OUTPUT_CLIP_PREFIX = "outputs/person{}_clip_city.mp4"       
DETECTIONS_CSV = "outputs/tracking_city.csv"       

model = YOLO(MODEL_PATH)

def draw_box(frame, x1, y1, x2, y2, pid):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"ID {pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def track_and_export(video_file, output_video, detections_csv):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    detections = []
    frame_no = 0
    print(f"Video info: {width}x{height}, {fps} FPS, {total} frames")
    start_time = time.time()
    print("Tracking started")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, verbose=False, tracker="botsort.yaml", conf=0.25,iou=0.5)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for i in range(len(ids)):
                cls = int(results[0].boxes.cls[i])
                if cls != 0: 
                    continue
                x1, y1, x2, y2 = map(int, boxes[i])
                pid = int(ids[i])
                frame = draw_box(frame, x1, y1, x2, y2, pid)
                detections.append({
                    "frame_no": frame_no,
                    "person_id": pid,
                    "confidence": round(float(confs[i]), 2),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
        out.write(frame)
        frame_no += 1
    cap.release()
    out.release()
    totaltime = time.time() - start_time
    print(f"Tracking complete - took {totaltime:.1f} seconds")
    print(f"Processed at {total/totaltime:.1f} fps")
    df = pd.DataFrame(detections)
    df.to_csv(detections_csv, index=False)
    print(f"Tracking results saved to {detections_csv}")

def export_clip(person_id, video_file, detections_csv, output_clip):
    cap = cv2.VideoCapture(video_file)
    df = pd.read_csv(detections_csv)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_clip, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frames_needed = set(df[df['person_id'] == person_id]['frame_no'])
    
    if len(frames_needed) == 0:
        print(f"Person ID {person_id} not found!")
        cap.release()
        out.release()
        return
    
    frame_no = 0
    print(f"Exporting clip for Person ID: {person_id}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in frames_needed:
            row = df[(df['frame_no'] == frame_no) & (df['person_id'] == person_id)].iloc[0]
            frame = draw_box(frame, int(row.x1), int(row.y1), int(row.x2), int(row.y2), person_id)
            out.write(frame)
        frame_no += 1
    cap.release()
    out.release()
    print(f"Exported clip: {output_clip}")


track_and_export(VIDEO_PATH, OUTPUT_TRACKED_VIDEO, DETECTIONS_CSV)

person_id = int(input("\nEnter Person ID to export clip: "))
export_clip(person_id, VIDEO_PATH, DETECTIONS_CSV, OUTPUT_CLIP_PREFIX.format(person_id))