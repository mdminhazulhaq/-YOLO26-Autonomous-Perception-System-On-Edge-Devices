import os
import pandas as pd
import numpy as np

# ==================================================
# LABEL PATHS
# ==================================================

train_dir = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\train"
val_dir   = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\val"

# ==================================================
# OUTPUT CSV
# ==================================================

output_csv = r"C:\Users\UM-User\Downloads\simple_distance_regression\distance_dataset_12features.csv"

# ==================================================
# READ LABELS
# ==================================================

data = []

for folder in [train_dir, val_dir]:

    for file in os.listdir(folder):

        if not file.endswith(".txt"):
            continue

        filepath = os.path.join(folder, file)

        with open(filepath, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 6:
                    continue

                class_id = float(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

                distance = float(parts[5])

                # ====================================
                # DERIVED FEATURES
                # ====================================

                bottom_y = y_center + (height / 2)

                area = width * height

                aspect_ratio = width / (height + 1e-6)

                diagonal = np.sqrt(
                    width**2 +
                    height**2
                )

                inverse_height = 1.0 / (height + 1e-6)

                inverse_area = 1.0 / (area + 1e-6)

                # confidence removed
                scale_score = diagonal

                data.append([
                    class_id,
                    x_center,
                    y_center,
                    bottom_y,
                    width,
                    height,
                    area,
                    aspect_ratio,
                    diagonal,
                    inverse_height,
                    inverse_area,
                    scale_score,
                    distance
                ])

# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame(
    data,
    columns=[
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
    output_csv,
    index=False
)

print("\nSaved:", output_csv)

print("Samples:", len(df))

print("\nColumns:")

for col in df.columns:
    print(col)