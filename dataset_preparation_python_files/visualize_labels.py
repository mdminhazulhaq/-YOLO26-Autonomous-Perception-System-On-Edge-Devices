import cv2
import os

image_path = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\images\train\frame_980.jpg"

label_path = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Dataset\labels\train\frame_980.txt"

img = cv2.imread(image_path)

h, w, _ = img.shape

with open(label_path, "r") as f:

    lines = f.readlines()

for line in lines:

    parts = line.strip().split()

    cls = int(parts[0])

    x = float(parts[1])
    y = float(parts[2])
    bw = float(parts[3])
    bh = float(parts[4])

    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)

    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

cv2.imshow("Validation", img)

cv2.waitKey(0)

cv2.destroyAllWindows()