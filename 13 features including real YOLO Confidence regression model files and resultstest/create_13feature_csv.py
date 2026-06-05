import os
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

# ==========================================================
# PATHS
# ==========================================================

YOLO_MODEL = r"C:\Users\UM-User\Downloads\ultralytics\runs\detect\yolo26n-300epochs\weights\best.pt"

TRAIN_IMAGES = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\images\train"
VAL_IMAGES   = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\images\val"

TRAIN_LABELS = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\train"
VAL_LABELS   = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\val"

OUTPUT_CSV = r"C:\Users\UM-User\Downloads\simple_distance_regression\distance_dataset_13features.csv"

# ==========================================================
# LOAD YOLO
# ==========================================================

model = YOLO(YOLO_MODEL)

# ==========================================================
# STORAGE
# ==========================================================

data = []

# ==========================================================
# PROCESS FUNCTION
# ==========================================================

def process_dataset(image_dir, label_dir):

    image_files = sorted(os.listdir(image_dir))

    for img_name in image_files:

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, img_name)

        label_path = os.path.join(
            label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        # ==================================================
        # LOAD GT LABELS
        # ==================================================

        gt_objects = []

        with open(label_path, "r") as f:

            for line in f:

                p = line.strip().split()

                if len(p) != 6:
                    continue

                gt_objects.append({

                    "class_id": int(float(p[0])),
                    "cx": float(p[1]),
                    "cy": float(p[2]),
                    "distance": float(p[5])

                })

        # ==================================================
        # YOLO DETECTION
        # ==================================================

        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h

                # ==========================================
                # MATCH GT
                # ==========================================

                best_gt = None
                best_dist = 999999

                for gt in gt_objects:

                    if gt["class_id"] != cls:
                        continue

                    d = np.sqrt(
                        (gt["cx"] - cx) ** 2 +
                        (gt["cy"] - cy) ** 2
                    )

                    if d < best_dist:

                        best_dist = d
                        best_gt = gt

                if best_gt is None:
                    continue

                distance = best_gt["distance"]

                # ==========================================
                # DERIVED FEATURES
                # ==========================================

                bottom_y = cy + bh / 2

                area = bw * bh

                aspect_ratio = bw / (bh + 1e-6)

                diagonal = np.sqrt(
                    bw ** 2 +
                    bh ** 2
                )

                inverse_height = 1.0 / (bh + 1e-6)

                inverse_area = 1.0 / (area + 1e-6)

                scale_score = diagonal * conf

                # ==========================================
                # SAVE
                # ==========================================

                data.append([

                    conf,
                    cls,

                    cx,
                    cy,
                    bottom_y,

                    bw,
                    bh,

                    area,
                    aspect_ratio,
                    diagonal,

                    inverse_height,
                    inverse_area,
                    scale_score,

                    distance

                ])

# ==========================================================
# RUN
# ==========================================================

print("Processing TRAIN...")
process_dataset(TRAIN_IMAGES, TRAIN_LABELS)

print("Processing VAL...")
process_dataset(VAL_IMAGES, VAL_LABELS)

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(

    data,

    columns=[

        "confidence",
        "class_id",

        "x_center",
        "y_center",
        "bottom_y",

        "width",
        "height",

        "area",
        "aspect_ratio",
        "diagonal",

        "inverse_height",
        "inverse_area",
        "scale_score",

        "distance"
    ]
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved:", OUTPUT_CSV)
print("Samples:", len(df))
print("\nColumns:")
print(df.columns.tolist())