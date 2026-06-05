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

DISTANCE_MODEL = r"C:\Users\UM-User\Downloads\simple_distance_regression\best_distance_model_12feature.pth"

SCALER_X = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_X_12feature.pkl"

SCALER_Y = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_y_12feature.pkl"

IMAGE_PATH = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\images\val\frame_50.jpg"

GT_LABEL = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\val\frame_50.txt"

# =====================================================
# REGRESSION MODEL
# =====================================================

class DistanceRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(12, 16),
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
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    0: "Ve",
    1: "Pr",
    2: "Cy",
    3: "Mo"
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 165, 255),
    3: (0, 0, 255)
}

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception("Image not found")

img_h, img_w = image.shape[:2]

# =====================================================
# LOAD GROUND TRUTH LABELS
# =====================================================

gt_objects = []

with open(GT_LABEL, "r") as f:

    for line in f:

        p = line.strip().split()

        if len(p) != 6:
            continue

        gt_objects.append({

            "cls": int(float(p[0])),
            "cx": float(p[1]),
            "cy": float(p[2]),
            "dist": float(p[5])

        })

# =====================================================
# YOLO DETECTION
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

        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h

        # =================================================
        # 12 FEATURES
        # =================================================

        bottom_y = cy + (bh / 2)

        area = bw * bh

        aspect_ratio = bw / (bh + 1e-6)

        diagonal = np.sqrt(
            bw ** 2 +
            bh ** 2
        )

        inverse_height = 1.0 / (bh + 1e-6)

        inverse_area = 1.0 / (area + 1e-6)

        scale_score = diagonal

        features = np.array([
            [
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
                scale_score
            ]
        ])

        features_scaled = scaler_X.transform(
            features
        )

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
        # GT MATCHING
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
        # DRAW BOX
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
            (int(x1), max(20, int(y1) - 10)),
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

output_path = "distance_result_12feature.jpg"

cv2.imwrite(
    output_path,
    image
)

print(f"\nSaved: {output_path}")