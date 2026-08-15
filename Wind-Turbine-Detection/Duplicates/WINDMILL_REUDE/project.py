import os
import random
import shutil
from ultralytics import YOLO

# ==============================
# CONFIG (EDIT IF NEEDED)
# ==============================
RAW_DATA = "NordTank586x371"
DATASET = "wind_turbine_dataset"
PREDICT_FOLDER = "predict"
BASE_MODEL = "yolov8n.pt"

TRAIN_RATIO = 0.8   # 80% train, 20% val

# ==============================
# CREATE DATASET STRUCTURE
# ==============================
for split in ["train", "val"]:
    os.makedirs(f"{DATASET}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET}/labels/{split}", exist_ok=True)

# ==============================
# READ IMAGES & CHECK LABELS
# ==============================
all_images = os.listdir(f"{RAW_DATA}/images")
all_images = [img for img in all_images if img.lower().endswith((".jpg", ".png", ".jpeg"))]

valid_images = []
skipped_images = []

for img in all_images:
    label_name = img.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(RAW_DATA, "labels", label_name)

    if os.path.exists(label_path):
        valid_images.append(img)
    else:
        skipped_images.append(img)

print("========== DATA CHECK ==========")
print(f"Total images found       : {len(all_images)}")
print(f"Images with labels       : {len(valid_images)}")
print(f"Images skipped (no label): {len(skipped_images)}")

# ==============================
# SHUFFLE & SPLIT
# ==============================
random.shuffle(valid_images)

split_index = int(len(valid_images) * TRAIN_RATIO)
train_images = valid_images[:split_index]
val_images = valid_images[split_index:]

print("========== SPLIT ==========")
print(f"Train images: {len(train_images)}")
print(f"Val images  : {len(val_images)}")

# ==============================
# COPY FILES
# ==============================
def copy_files(image_list, split):
    for img in image_list:
        # copy image
        shutil.copy(
            os.path.join(RAW_DATA, "images", img),
            os.path.join(DATASET, "images", split, img)
        )

        # copy label
        label = img.rsplit(".", 1)[0] + ".txt"
        shutil.copy(
            os.path.join(RAW_DATA, "labels", label),
            os.path.join(DATASET, "labels", split, label)
        )

copy_files(train_images, "train")
copy_files(val_images, "val")

print("✅ Dataset split completed")

# ==============================
# CREATE data.yaml (2 CLASSES)
# ==============================
with open("data.yaml", "w") as f:
    f.write(
        "path: wind_turbine_dataset\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 2\n"
        "names:\n"
        "  - damage_type_0\n"
        "  - damage_type_1\n"
    )

print("✅ data.yaml created (nc=2)")

# ==============================
# TRAIN YOLO
# ==============================
model = YOLO(BASE_MODEL)

model.train(
    data="data.yaml",
    epochs=5,
    imgsz=1200,
    batch=8
)

print("✅ Training completed")

# ==============================
# PREDICT ON NEW IMAGES
# ==============================
trained_model = YOLO("runs/detect/train/weights/best.pt")

trained_model.predict(
    source=PREDICT_FOLDER,
    conf=0.25,
    save=True
)

print("✅ Predictions completed")
