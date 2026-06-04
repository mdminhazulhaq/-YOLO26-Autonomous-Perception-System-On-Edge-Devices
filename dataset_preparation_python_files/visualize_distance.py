import os
import cv2

# =========================================================
# PATHS
# =========================================================

image_folder = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\images\train"

distance_folder = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\distance_labels\train"

# =========================================================
# SELECT FRAME
# =========================================================

frame_name = "frame_100"

image_path = os.path.join(image_folder, frame_name + ".jpg")

label_path = os.path.join(distance_folder, frame_name + ".txt")

# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(image_path)

height, width, _ = image.shape

# =========================================================
# READ LABELS
# =========================================================

with open(label_path, "r") as f:
    lines = f.readlines()

# =========================================================
# DRAW BOXES
# =========================================================

for line in lines:

    values = line.strip().split()

    cls = int(values[0])

    x_center = float(values[1])
    y_center = float(values[2])

    box_width = float(values[3])
    box_height = float(values[4])

    distance = float(values[5])

    # =====================================================
    # YOLO -> PIXEL
    # =====================================================

    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)

    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)

    # =====================================================
    # DRAW BOX
    # =====================================================

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # =====================================================
    # DRAW DISTANCE
    # =====================================================

    text = f"{distance:.1f}m"

    cv2.putText(
        image,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

# =========================================================
# SHOW IMAGE
# =========================================================

cv2.imshow("Distance Validation", image)

cv2.waitKey(0)

cv2.destroyAllWindows()