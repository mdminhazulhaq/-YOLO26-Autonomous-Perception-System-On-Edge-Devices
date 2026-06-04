import os
import random
import cv2

# =====================================================
# PATHS
# =====================================================

image_folder = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\images\train"

label_folder = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\distance_labels\train"

# =====================================================
# RANDOM IMAGE SELECTION
# =====================================================

all_images = [
    f for f in os.listdir(image_folder)
    if f.endswith(".jpg")
]

random_images = random.sample(all_images, 5)

print("\nRANDOM IMAGES:\n")

for img_name in random_images:
    print(img_name)

# =====================================================
# VISUALIZATION
# =====================================================

for img_name in random_images:

    image_path = os.path.join(
        image_folder,
        img_name
    )

    label_path = os.path.join(
        label_folder,
        img_name.replace(".jpg", ".txt")
    )

    image = cv2.imread(image_path)

    height, width, _ = image.shape

    # =================================================
    # READ LABELS
    # =================================================

    with open(label_path, "r") as f:
        lines = f.readlines()

    # =================================================
    # PROCESS EACH OBJECT
    # =================================================

    for line in lines:

        values = line.strip().split()

        if len(values) < 6:
            continue

        # =================================================
        # LABEL VALUES
        # =================================================

        cls = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])

        box_width = float(values[3])
        box_height = float(values[4])

        distance = float(values[5])

        # =================================================
        # REMOVE VERY SMALL NOISY BOXES
        # =================================================

        if box_width < 0.01 or box_height < 0.01:
            continue

        # =================================================
        # YOLO TO PIXEL COORDINATES
        # =================================================

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)

        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # =================================================
        # CLASS COLORS + NAMES
        # =================================================

        if cls == 0:

            class_name = "Vehicle"

            color = (0, 255, 0)      # GREEN

        elif cls == 1:

            class_name = "Pedestrian"

            color = (255, 0, 0)      # BLUE

        elif cls == 2:

            class_name = "Cyclist"

            color = (0, 255, 255)    # YELLOW

        else:

            class_name = "Unknown"

            color = (255, 255, 255)  # WHITE

        # =================================================
        # DISPLAY TEXT
        # =================================================

        text = f"{class_name} {distance:.1f}m"

        # =================================================
        # DRAW BOUNDING BOX
        # =================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # =================================================
        # DRAW TEXT
        # =================================================

        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # =================================================
    # SHOW IMAGE
    # =================================================

    cv2.imshow(img_name, image)

# =====================================================
# WAIT
# =====================================================

cv2.waitKey(0)

cv2.destroyAllWindows()