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
#### Waymo Open Dataset Github public repo: 
https://github.com/waymo-research/waymo-open-dataset/tree/master

#### Tuterial in Collab:
https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb

#### The dataset was prepared using the 
Waymo Open Dataset v2.0.1. Link:
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

##### After the images in .jpg format extracted inside the train folder and labels txt files in the lebels folder, we splited the images and labels into 80:20 for train: validation ratio for traning on YOLO26 model.


YOLO label format:

```text
class_id x_center y_center width height
```

Distance label format:

```text
class_id x_center y_center width height distance
```

### Distance Calculation

The LiDAR box parquet file contains the 3D center coordinates of each detected object:

key.laser_object_id = 8f3c...

[LiDARBoxComponent].box.center.x = 32.4
[LiDARBoxComponent].box.center.y = -4.1
[LiDARBoxComponent].box.center.z = 1.8

where:

x = forward distance (m)
y = left/right offset (m)
z = height (m)

The Euclidean distance from the ego vehicle to the object is calculated as:

Distance = √(x² + y² + z²)

Example:

x = 32.4
y = -4.1
z = 1.8
Distance = √(32.4² + (-4.1)² + 1.8²)
         = √(1049.76 + 16.81 + 3.24)
         = √1069.81
         = 32.71 m

The computed distance is then matched with the corresponding projected LiDAR bounding box and stored in the distance label file:

class_id x_center y_center width height distance

Example:

0 0.523 0.614 0.125 0.181 32.71

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

### The prepared dataset are stored in the drive in zip file format. Get the access of the dataset in the link:

https://drive.google.com/file/d/1ZlCuwDcWz9nEeoWZBC8AY3y2EdtlNK-u/view?usp=sharing

## 2nd Stage: Waymo_YOLO_Distance dataset traning on YOLO26n model for object detection

### Installed Prerequisites
* Python 3.10
* PyTorch 2.7
* CUDA 11.8
* Ultralytics

### Hardware
* Windows 11 with NVIDIA RTX A2000 Laptop GPU (4 GB VRAM)

### Install Ultralytics

```bash
pip install ultralytics
```

### Training Command

The YOLO26n model was trained for 300 epochs using the prepared Waymo_YOLO_Distance dataset:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo26n.yaml data=C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\dataset.yaml epochs=300 imgsz=640 batch=4 device=0 workers=0 optimizer=AdamW lr0=0.001 weight_decay=0.0005 patience=30 cache=False plots=True name=yolo26n-300epochs
```

### Training Results
After 300 epochs traning, the traning results

![YOLO26n Training Results](yolo26n_300epochs_training_results_screenshoot.png)

![YOLO26n_Performances_and_Loss](results.png)

The best-performing model (`best.pt`) will use for the distance estimation stage.

## Stage 3: Distance Regression Model Development and Optimization

### Regression Model

A lightweight feed-forward neural network was developed using PyTorch to estimate object distance from object detection features.

```python
class DistanceRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(13, 16),
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
```

### Model Configuration

| Component              | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| Input Layer            | Feature vector extracted from object detection results |
| Hidden Layers          | 16 → 32 → 16 neurons                                   |
| Activation Function    | ReLU                                                   |
| Dropout                | 0.2                                                    |
| Output Layer           | Single distance value (meters)                         |
| Loss Function          | Mean Squared Error (MSE)                               |
| Optimizer              | Adam                                                   |
| Learning Rate          | 0.001                                                  |
| Epochs                 | 100                                                    |
| Train/Validation Split | 90% / 10%                                              |

### Loss Function

The model was trained using Mean Squared Error (MSE):

```text
MSE = (1/N) Σ (Predicted Distance − Ground Truth Distance)²
```

The loss penalizes large distance prediction errors more heavily and guides the model toward accurate distance estimation.

---

### CSV Dataset Generation

Distance labels and object detection features were converted into CSV format for regression model training.

#### The regression model was tranined and evaluated on the different number of features dataset. First we trained the model on 5 features csv dataest, this dataset is stored in folder "5 feature regression model files and results" and named 'distance_dataset_5cols.csv'. Then we trained and validated on 6 features, 8 features, 12 features and finally 13 features csv datasets. For ecah experiment the dataset is splited into 90% train and 10 % validation. In the experiment, the target feature is 'distance' for each model. So, the model will liearn and train on the other features and calculate the distance. 

| 5 Features | 6 Features | 8 Features   | 12 Features    | 13 Features    |
| ---------- | ---------- | ------------ | -------------- | -------------- |
| x_center   | class_id   | x_center     | class_id       | confidence     |
| y_center   | x_center   | y_center     | x_center       | class_id       |
| width      | y_center   | width        | y_center       | x_center       |
| height     | width      | height       | bottom_y       | y_center       |
| distance   | height     | bottom_y     | width          | bottom_y       |
|            | distance   | area         | height         | width          |
|            |            | aspect_ratio | area           | height         |
|            |            | diagonal     | aspect_ratio   | area           |
|            |            | distance     | diagonal       | aspect_ratio   |
|            |            |              | inverse_height | diagonal       |
|            |            |              | inverse_area   | inverse_height |
|            |            |              | scale_score    | inverse_area   |
|            |            |              | distance       | scale_score    |
|            |            |              |                | distance       |

#### Derived Features, x_center, y_center, width, height, distance already we have form the dataset and the other features we calculated using the following formulas.

```text
bottom_y       = y_center + height / 2
area           = width × height
aspect_ratio   = width / height
diagonal       = √(width² + height²)
inverse_height = 1 / height
inverse_area   = 1 / area
scale_score    = diagonal × confidence
```

#### Feature Progression

```text
5 Features  → Basic bounding box geometry
6 Features  → Added object class information
8 Features  → Added geometric and scale features
12 Features → Added depth-sensitive features
13 Features → Added YOLO detection confidence
```

### Regression Model Ablation Study

| Features    | Training Samples | Parameters |   MAE (m) | RMSE (m) |
| ----------- | ---------------: | ---------: | --------: | -------: |
| 5 Features  |           13,327 |      1,169 |     4.763 |    6.511 |
| 6 Features  |           13,327 |      1,185 |     4.769 |    6.457 |
| 8 Features  |           13,327 |      1,233 |     4.803 |    6.540 |
| 12 Features |           13,327 |      1,297 |     4.137 |    5.564 |
| 13 Features |           12,532 |      1,313 | **3.905** |    5.728 |

### Image-Level Evaluation

The trained YOLO26n detector and regression model were jointly evaluated on Waymo validation images.

| Features    | Image MAE (m) | Image RMSE (m) |
| ----------- | ------------: | -------------: |
| 5 Features  |         6.256 |          7.915 |
| 6 Features  |         6.776 |          8.151 |
| 8 Features  |         6.432 |          8.175 |
| 12 Features |     **5.975** |          7.842 |
| 13 Features |         6.126 |      **7.619** |

The 12-feature model achieved the best image-level MAE, while the 13-feature model achieved the lowest image-level RMSE.




### To compare the GT with Predicted Distance, we run the 'compare_simple_prediction' file, which we will get inside the folder '13 features including real YOLO Confidence regression model files and resultstest' folder
Anaconda prompt run command

(yolov8) C:\Users\UM-User\Downloads\simple_distance_regression>python compare_simple_prediction.py


```python
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

DISTANCE_MODEL = r"C:\Users\UM-User\Downloads\simple_distance_regression\best_distance_model_13feature.pth"

SCALER_X = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_X_13feature.pkl"

SCALER_Y = r"C:\Users\UM-User\Downloads\simple_distance_regression\scaler_y_13feature.pkl"

IMAGE_PATH = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\images\val\frame_50.jpg"

GT_LABEL = r"C:\Users\UM-User\Downloads\Waymo_YOLO_Distance\labels-with-distances\val\frame_50.txt"

# =====================================================
# REGRESSION MODEL
# =====================================================

class DistanceRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(13, 16),
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
# LOAD GT LABELS
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

    for box in result.boxes:

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h

        # =================================================
        # 13 FEATURES
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

        scale_score = diagonal * conf

        features = np.array([
            [
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
        # DISPLAY LABEL
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

output_path = "distance_result_13feature.jpg"

cv2.imwrite(
    output_path,
    image
)

print(f"\nSaved: {output_path}")
```







### 5-Feature Regression Model: Ground Truth vs Predicted Distance with YOLO26n Detection
![YOLO26n Object Detection with bounding box score + Regression model distance prediciton with GT distance comparison](distance_result_5_features.jpg)



![YOLO26n Object Detection with bounding box score + Regression model distance prediciton with GT distance comparison](result_for_5_features_frame_50.jpg)


















