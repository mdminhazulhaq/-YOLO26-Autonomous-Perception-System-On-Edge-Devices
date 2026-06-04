# YOLO26-Autonomous-Perception-System-On-Edge-Devices

Autonomous driving perception framework for object detection and monocular distance estimation using YOLO26 and a geometry-aware distance regression model.

## Summary of the study
#### 1st Stage: Dataset Preparation
#### 2nd Stage: YOLO26 model traning and validation on the prepared dataset
#### 3rd Stage: Regression model developed, tranined & validated, optimzed untile get satisfactory results
#### 4th Stage: Run integrated (YOLO26 + Regression) model for object detection and distance calculation
#### 5th Stage: Optimize YOLO26 model for better trade off between accuracy and inference speed
#### Final Stage: Deploy and run the model on Jetson Thor platforms


## 1st Stage: Dataset Preparation

The dataset was prepared using the Waymo Open Dataset v2.0.1.

Dataset Link:

https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1

### Initial Parquet Files

The following parquet files were used:

* training_camera_image_10023947602400723454_1120_000_1140_000.parquet
* training_projected_lidar_box_10023947602400723454_1120_000_1140_000.parquet
* training_lidar_box_10023947602400723454_1120_000_1140_000.parquet

### Step 1: Dataset Exploration

#### All python files for dataset preparation and validation are located inside the dataset_preparation_python_files folder in the repository

Script:

```text
extract_waymo.py
```

Purpose:

* Read LiDAR parquet data
* Inspect available columns and records

### Step 2: Image, Label and Distance Extraction

Script:

```text
final_waymo_extractor.py
```

Purpose:

* Extract RGB images from camera image parquet files
* Generate YOLO bounding box labels
* Generate distance labels using LiDAR annotations

Generated files:

```text
images/train/
labels/train/
distance_labels/train/
```

YOLO label format:

```text
class_id x_center y_center width height
```

Distance label format:

```text
class_id x_center y_center width height distance
```

### Distance Calculation

The LiDAR box file provides 3D object center coordinates:

* x = forward distance
* y = lateral distance
* z = height

Distance is calculated as:

```text
Distance = √(x² + y² + z²)
```

### Step 3: Label Verification

Scripts:

```text
visualize_labels.py
visualize_distance.py
random_visualization.py
```

Purpose:

* Verify bounding boxes
* Verify object distances
* Perform random dataset inspection

### Step 4: Final Dataset Generation

Script:

```text
create_dataset_structure.py
```

Purpose:

* Remove invalid samples
* Verify image-label consistency
* Create training and validation sets
* Generate dataset.yaml

### Final Dataset Statistics

```text
Training Images     : 794
Validation Images  : 199
Total Images       : 993
```

### Final Directory Structure

```text
Waymo_YOLO_Distance/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── distance_labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```
