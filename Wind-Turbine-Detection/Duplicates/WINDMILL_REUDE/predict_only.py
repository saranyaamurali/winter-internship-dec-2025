from ultralytics import YOLO

model = YOLO("runs/detect/train_1200/weights/best.pt")

model.predict(
    source="predict",
    conf=0.1,
    save=True
)
