import os
import shutil
import random
from sklearn.model_selection import train_test_split

# =========================================================
# ORIGINAL DATA PATHS
# =========================================================

IMAGE_DIR = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\images\train"

LABEL_DIR = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\labels\train"

DISTANCE_DIR = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\distance_labels\train"

# =========================================================
# FINAL DATASET ROOT
# =========================================================

OUTPUT_ROOT = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance"

# =========================================================
# CREATE FINAL FOLDERS
# =========================================================

folders = [

    os.path.join(OUTPUT_ROOT, "images", "train"),
    os.path.join(OUTPUT_ROOT, "images", "val"),

    os.path.join(OUTPUT_ROOT, "labels", "train"),
    os.path.join(OUTPUT_ROOT, "labels", "val"),

    os.path.join(OUTPUT_ROOT, "distance_labels", "train"),
    os.path.join(OUTPUT_ROOT, "distance_labels", "val"),
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("\nFOLDERS CREATED!\n")

# =========================================================
# GET ALL IMAGE FILES
# =========================================================

all_images = [

    f for f in os.listdir(IMAGE_DIR)

    if f.endswith(".jpg")
]

print(f"TOTAL IMAGES FOUND: {len(all_images)}")

# =========================================================
# CHECK VALID FILES
# =========================================================

valid_images = []

missing_detection = 0
missing_distance = 0

for img_name in all_images:

    label_file = img_name.replace(".jpg", ".txt")

    detection_path = os.path.join(
        LABEL_DIR,
        label_file
    )

    distance_path = os.path.join(
        DISTANCE_DIR,
        label_file
    )

    if not os.path.exists(detection_path):

        missing_detection += 1
        continue

    if not os.path.exists(distance_path):

        missing_distance += 1
        continue

    # =====================================================
    # CHECK EMPTY FILES
    # =====================================================

    if os.path.getsize(detection_path) == 0:
        continue

    if os.path.getsize(distance_path) == 0:
        continue

    valid_images.append(img_name)

print(f"\nVALID IMAGES: {len(valid_images)}")
print(f"MISSING DETECTION LABELS: {missing_detection}")
print(f"MISSING DISTANCE LABELS: {missing_distance}")

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_images, val_images = train_test_split(
    valid_images,
    test_size=0.2,
    random_state=42
)

print(f"\nTRAIN IMAGES: {len(train_images)}")
print(f"VAL IMAGES: {len(val_images)}")

# =========================================================
# COPY FUNCTION
# =========================================================

def copy_dataset(files, split_name):

    for img_name in files:

        label_name = img_name.replace(".jpg", ".txt")

        # =================================================
        # SOURCE PATHS
        # =================================================

        src_img = os.path.join(
            IMAGE_DIR,
            img_name
        )

        src_label = os.path.join(
            LABEL_DIR,
            label_name
        )

        src_distance = os.path.join(
            DISTANCE_DIR,
            label_name
        )

        # =================================================
        # DESTINATION PATHS
        # =================================================

        dst_img = os.path.join(
            OUTPUT_ROOT,
            "images",
            split_name,
            img_name
        )

        dst_label = os.path.join(
            OUTPUT_ROOT,
            "labels",
            split_name,
            label_name
        )

        dst_distance = os.path.join(
            OUTPUT_ROOT,
            "distance_labels",
            split_name,
            label_name
        )

        # =================================================
        # COPY FILES
        # =================================================

        shutil.copy(src_img, dst_img)

        shutil.copy(src_label, dst_label)

        shutil.copy(src_distance, dst_distance)

# =========================================================
# COPY TRAIN
# =========================================================

print("\nCOPYING TRAIN DATA...\n")

copy_dataset(
    train_images,
    "train"
)

# =========================================================
# COPY VALIDATION
# =========================================================

print("\nCOPYING VALIDATION DATA...\n")

copy_dataset(
    val_images,
    "val"
)

# =========================================================
# CREATE dataset.yaml
# =========================================================

yaml_path = os.path.join(
    OUTPUT_ROOT,
    "dataset.yaml"
)

yaml_text = f"""
path: {OUTPUT_ROOT}

train: images/train
val: images/val

nc: 3

names:
  0: Vehicle
  1: Pedestrian
  2: Cyclist
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("\ndataset.yaml CREATED!")

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n==============================")
print("FINAL DATASET READY!")
print("==============================")

print(f"\nTOTAL VALID IMAGES: {len(valid_images)}")
print(f"TRAIN: {len(train_images)}")
print(f"VAL: {len(val_images)}")

print("\nDATASET LOCATION:")
print(OUTPUT_ROOT)