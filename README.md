# YOLO26-Autonomous-Perception-System-On-Edge-Devices
Autonomous driving perception framework for object detection and monocular distance estimation using YOLO26 and geometry-aware distance regression.

Stage 1: Dataset Preparation

The dataset was prepared using the Waymo Open Dataset. Four parquet files were utilized:

training_camera_image_1_10023947602400723454_1120_000_1140_000.parquet
training_camera_box_1_10023947602400723454_1120_000_1140_000.parquet
training_camera_to_lidar_box_association_1_10023947602400723454_1120_000_1140_000.parquet
training_lidar_box_1_10023947602400723454_1120_000_1140_000.parquet
Image Extraction

Camera images were extracted from the camera image parquet file. The images were stored as JPEG-encoded bytes and decoded using OpenCV. A total of 995 RGB images were extracted and saved in JPG format.

Bounding Box Label Generation

Bounding box annotations were obtained from the camera box parquet file. The annotations were converted into YOLO format:

class_id x_center y_center width height

Vehicle and pedestrian classes were retained for training.

Distance Label Generation

The camera-to-LiDAR association file was used to match camera objects with LiDAR objects. The LiDAR box file provided the 3D object center coordinates:

x: forward distance
y: lateral distance
z: height

The object distance was computed as:

Distance = √(x² + y² + z²)

The generated labels were stored as:

class_id x_center y_center width height distance
Dataset Split

The final dataset contained:

Training images: 794
Validation images: 199
Directory Structure
images/
├── train/
└── val/

labels/
├── train/
└── val/

labels-with-distances/
├── train/
└── val/
