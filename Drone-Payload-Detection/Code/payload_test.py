import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "container_detection\yolo11_images_train\weights/best.pt"   # or best.pt
VIDEO_PATH = "RC5_OR_0004.mp4" # 0 for webcam
CONF_THRESH = 0.4
IMG_SIZE = 640

FRAME_SKIP = 2

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video source")

# -----------------------------
# VIDEO PROPERTIES
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Input FPS: {fps}")

# -----------------------------
# OUTPUT VIDEO (2× FPS)
# -----------------------------
out = cv2.VideoWriter(
    "output_detected_2x.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps * 2,   # 🔥 double FPS
    (width, height)
)

# -----------------------------
# INFERENCE LOOP
# -----------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 SKIP FRAMES (REAL SPEED BOOST)
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )

    annotated = results[0].plot()

    # YOLO gives RGB → OpenCV needs BGR
    annotated = annotated[:, :, ::-1]

    out.write(annotated)
    cv2.imshow("Container Detection (2x Speed)", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Saved as output_detected_2x.mp4")
