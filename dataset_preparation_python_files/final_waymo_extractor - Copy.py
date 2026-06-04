import os
import math
import pandas as pd
from PIL import Image
import io

# =========================================================
# FILE PATHS
# =========================================================

base_path = r"C:\Users\UM-User\Downloads"

camera_image_path = os.path.join(
    base_path,
    "training_camera_image_10023947602400723454_1120_000_1140_000.parquet"
)

projected_box_path = os.path.join(
    base_path,
    "training_projected_lidar_box_10023947602400723454_1120_000_1140_000.parquet"
)

lidar_box_path = os.path.join(
    base_path,
    "training_lidar_box_10023947602400723454_1120_000_1140_000.parquet"
)

# =========================================================
# OUTPUT PATHS
# =========================================================

image_output = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\images\train"

label_output = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\labels\train"

distance_output = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\distance_labels\train"

os.makedirs(image_output, exist_ok=True)
os.makedirs(label_output, exist_ok=True)
os.makedirs(distance_output, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

camera_df = pd.read_parquet(camera_image_path)

projected_df = pd.read_parquet(projected_box_path)

lidar_df = pd.read_parquet(lidar_box_path)

print("Camera Images:", len(camera_df))
print("Projected Boxes:", len(projected_df))
print("LiDAR Boxes:", len(lidar_df))

# =========================================================
# CREATE DISTANCE LOOKUP
# =========================================================

lidar_lookup = {}

for _, row in lidar_df.iterrows():

    laser_id = row['key.laser_object_id']

    x = row['[LiDARBoxComponent].box.center.x']
    y = row['[LiDARBoxComponent].box.center.y']
    z = row['[LiDARBoxComponent].box.center.z']

    distance = math.sqrt(x*x + y*y + z*z)

    lidar_lookup[laser_id] = distance

print("LiDAR distance lookup created!")

# =========================================================
# PROCESS IMAGES
# =========================================================

frame_count = 0

for _, img_row in camera_df.iterrows():

    try:

        image_bytes = img_row['[CameraImageComponent].image']

        image = Image.open(io.BytesIO(image_bytes))

        width, height = image.size

        timestamp = img_row['key.frame_timestamp_micros']

        camera_name = img_row['key.camera_name']

        frame_name = f"frame_{frame_count}"

        image_save_path = os.path.join(
            image_output,
            frame_name + ".jpg"
        )

        image.save(image_save_path)

        # =====================================================
        # MATCH PROJECTED BOXES
        # =====================================================

        matched_boxes = projected_df[
            (projected_df['key.frame_timestamp_micros'] == timestamp) &
            (projected_df['key.camera_name'] == camera_name)
        ]

        yolo_lines = []

        distance_lines = []

        # =====================================================
        # PROCESS OBJECTS
        # =====================================================

        for _, box in matched_boxes.iterrows():

            object_type = box['[ProjectedLiDARBoxComponent].type']

            # =================================================
            # CLASS MAPPING
            # =================================================

            if object_type == 1:
                class_id = 0      # Vehicle

            elif object_type == 2:
                class_id = 1      # Pedestrian

            elif object_type == 4:
                class_id = 2      # Cyclist

            else:
                continue

            # =================================================
            # LASER ID
            # =================================================

            laser_id = box['key.laser_object_id']

            # =================================================
            # DISTANCE
            # =================================================

            distance = 0.0

            if laser_id in lidar_lookup:
                distance = lidar_lookup[laser_id]

            # =================================================
            # 2D BOX
            # =================================================

            cx = box['[ProjectedLiDARBoxComponent].box.center.x']
            cy = box['[ProjectedLiDARBoxComponent].box.center.y']

            bw = box['[ProjectedLiDARBoxComponent].box.size.x']
            bh = box['[ProjectedLiDARBoxComponent].box.size.y']

            # =================================================
            # NORMALIZE
            # =================================================

            x_center = cx / width
            y_center = cy / height

            box_width = bw / width
            box_height = bh / height

            # =================================================
            # FILTER INVALID BOXES
            # =================================================

            if box_width <= 0 or box_height <= 0:
                continue

            if box_width > 1 or box_height > 1:
                continue

            # =================================================
            # YOLO LABEL
            # =================================================

            yolo_line = (
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{box_width} "
                f"{box_height}"
            )

            yolo_lines.append(yolo_line)

            # =================================================
            # DISTANCE LABEL
            # =================================================

            distance_line = (
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{box_width} "
                f"{box_height} "
                f"{distance}"
            )

            distance_lines.append(distance_line)

        # =====================================================
        # SAVE YOLO LABELS
        # =====================================================

        label_path = os.path.join(
            label_output,
            frame_name + ".txt"
        )

        with open(label_path, "w") as f:

            for line in yolo_lines:
                f.write(line + "\n")

        # =====================================================
        # SAVE DISTANCE LABELS
        # =====================================================

        distance_path = os.path.join(
            distance_output,
            frame_name + ".txt"
        )

        with open(distance_path, "w") as f:

            for line in distance_lines:
                f.write(line + "\n")

        print(f"Processed: {frame_name}")

        frame_count += 1

    except Exception as e:

        print("ERROR:", e)

print("\nDONE!")
print("Total Frames:", frame_count)