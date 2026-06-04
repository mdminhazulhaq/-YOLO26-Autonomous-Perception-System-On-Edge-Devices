import cv2
import torch
import joblib
import numpy as np
from ultralytics import YOLO
import torch.nn as nn

# =====================================================
# PATHS
# =====================================================

YOLO_MODEL = r"C:\Users\UM-User\Downloads\ultralytics\runs\detect\yolo26n-300epochs\weights\best.pt"

DISTANCE_MODEL = r"C:\Users\UM-User\Downloads\simple_distance_regression\best_distance_model.pth"

SCALER_X = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_X.pkl"

SCALER_Y = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_y.pkl"

IMAGE_PATH = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\images\val\frame_50.jpg"

GT_LABEL = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\val\frame_50.txt"

# =====================================================
# DISTANCE MODEL
# =====================================================

class DistanceRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)

        )

    def forward(self, x):
        return self.network(x)

# =====================================================
# LOAD MODELS
# =====================================================

yolo = YOLO(YOLO_MODEL)

distance_model = DistanceRegressionModel()

distance_model.load_state_dict(
    torch.load(
        DISTANCE_MODEL,
        map_location="cpu"
    )
)

distance_model.eval()

scaler_X = joblib.load(SCALER_X)
scaler_y = joblib.load(SCALER_Y)

# =====================================================
# CLASS SETTINGS
# =====================================================

CLASS_NAMES = {
    0: "Ve",
    1: "Pr",
    2: "Cy",
    3: "Mo"
}

CLASS_COLORS = {
    0: (0, 255, 0),      # Green
    1: (255, 0, 0),      # Blue
    2: (0, 165, 255),    # Orange
    3: (0, 0, 255)       # Red
}

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception("Image not found")

h, w = image.shape[:2]

# =====================================================
# LOAD GT LABELS
# =====================================================

gt_objects = []

with open(GT_LABEL, "r") as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) != 6:
            continue

        gt_objects.append({
            "cls": int(float(parts[0])),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "dist": float(parts[5])
        })

# =====================================================
# YOLO INFERENCE
# =====================================================

results = yolo.predict(
    source=IMAGE_PATH,
    conf=0.25,
    verbose=False
)

print("\nGT vs Prediction\n")

errors = []

for result in results:

    boxes = result.boxes

    for box in boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h

        # =================================================
        # 5-FEATURE DISTANCE PREDICTION
        # =================================================

        features = np.array([
            [
                cls,
                cx,
                cy,
                bw,
                bh
            ]
        ])

        features_scaled = scaler_X.transform(features)

        features_tensor = torch.tensor(
            features_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():

            pred_scaled = distance_model(
                features_tensor
            )

        pred_distance = scaler_y.inverse_transform(
            pred_scaled.numpy()
        )[0][0]

        # =================================================
        # MATCH GT
        # =================================================

        best_gt = None
        best_match = 999999

        for gt in gt_objects:

            if gt["cls"] != cls:
                continue

            d = np.sqrt(
                (gt["cx"] - cx) ** 2 +
                (gt["cy"] - cy) ** 2
            )

            if d < best_match:

                best_match = d
                best_gt = gt

        gt_distance = 0
        error = 0

        if best_gt is not None:

            gt_distance = best_gt["dist"]

            error = abs(
                pred_distance -
                gt_distance
            )

            errors.append(error)

        # =================================================
        # LABEL TEXT
        # =================================================

        short_name = CLASS_NAMES.get(
            cls,
            f"C{cls}"
        )

        label = (
            f"{short_name}:{conf:.2f} "
            f"GT:{gt_distance:.1f}m "
            f"PD:{pred_distance:.1f}m "
            f"Er:{error:.1f}m"
        )

        color = CLASS_COLORS.get(
            cls,
            (255, 0, 255)
        )

        # =================================================
        # DRAW
        # =================================================

        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

        cv2.putText(
            image,
            label,
            (int(x1), max(20, int(y1)-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

        print(label)

# =====================================================
# FINAL METRICS
# =====================================================

if len(errors):

    mae = np.mean(errors)

    rmse = np.sqrt(
        np.mean(
            np.square(errors)
        )
    )

    print("\n===================")
    print("FINAL RESULTS")
    print("===================")
    print(f"MAE  : {mae:.3f} m")
    print(f"RMSE : {rmse:.3f} m")

# =====================================================
# SAVE IMAGE
# =====================================================

output_path = "distance_result.jpg"

cv2.imwrite(
    output_path,
    image
)

print(f"\nSaved: {output_path}")