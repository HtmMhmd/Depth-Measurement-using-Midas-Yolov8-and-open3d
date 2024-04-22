# Wooden Box Measurement

This project aims to measure the dimensions of a wooden box using iamge processing and AI detection techniques like yolo and harris corner detection. The measurements will be used for accurate planning and designing purposes.
i used in the project custom data annotated using **RoboFlow**, detected using **YoloV8** and generated 3d model using **Midas**.

## Table of Contents

## Table of Contents

[Wooden Box Measurement](#wooden-box-measurement)
- [Project Overview](#project-overview)
    - [Tools and Equipment](#tools-and-equipment)
    - [Data Collection](#data-collection)
    - [Analysis and Reporting](#analysis-and-reporting)
- [Getting Started](#getting-started)
- [Code Example](#code-example)
    - [1. Yolov8n training](#1-yolov8n-training)
    - [2. Corner detection using WoodenBox_detector Package](#2-corner-detection-using-woodenbox_detector-package)
    - [3. 3D model generation](#3-3d-model-generation)
        

---

## Project Overview

The wooden box measurement project consists of the following components:

### 1. **Tools and Equipment**: List the tools and equipment required for accurate measurements, including the specific models or brands if applicable.

- **Wooden box**: Used as the object to measure the dimensions.
- **Meter**: Used to measure the distance of the wooden box from the camera.
- **RoboFlow**: Used for data annotation.
- **YoloV8 nano**: Used for box detection.
- **Midas** hybrid: Used for box 3d model.

### 2. **Data Collection**: 

Collect custom data of the wooden box for training the AI detection model. This can be done by capturing images of the wooden box from different angles and perspectives. Ensure that the images cover a wide range of variations in lighting conditions, backgrounds, and box orientations. Annotate the images with bounding boxes around the wooden box using a tool like RoboFlow. This annotated data will be used to train the YOLOv8 model for detecting the wooden box accurately.

### 3. **Analysis and Reporting**: 

In the initial phase of the project, I used 5 collected pictures to train both the **YOLO Nano** model and the **YOLO Medium** model. However, the performance of the models was not satisfactory even after tuning. To improve the performance, I increased the data to 18 images and retrained both models. After tuning and testing, the **YOLO Nano** model showed good performance on both the testing data and in real-time scenarios.

The collected measurement data will be further analyzed to generate accurate 2D or 3D models of the wooden box. This analysis will help in accurate planning and designing purposes.

---
## Getting Started

To get started with the custom object measurement project, follow these steps:

1. Collect data of known dimentions object in different lightening conditions(prefered to be equal dimentions).

2. Annotate the desired object using [RoboFlow](https://universe.roboflow.com/hatem-mohamed-ygb1n/box-js3ok).

3. YoloV8 model [training](YoloModelFiles\Wooden_Box_YoloV8_Train.ipynb).

4. Object Boundary detection using Harris Corner detection.

5. Train Linear Regression model on the ground truth data and the detected box boundaries.

6. Test on Runtime.

---
## Code Example

### 1. Yolov8n training
The preformance of yolov8n on training data.
![Image got from yolo tarining files](YoloModelFiles\train_batch272.jpg)

**Using adamW as an Optmizer and lr0 of 0.0000001 for 150 epochs with early stopping i have managed to have mAP50 0.771 and mAP50-95 0.73847 at last epoch.**

full trainig verbose can be viewed from [here](YoloModelFiles\results.csv).
```python
results = model.train(data='data.yaml',
                      epochs=100, imgsz=640, batch=16, workers=8, dropout=0.5, augment=True, optimizer='AdamW', lr0=0.0000001)
```
The model [best.pt](YoloModelFiles\best.pt) along with [the training notebook](YoloModelFiles\Wooden_Box_YoloV8_Train.ipynb) file is attached in [YoloModelFiles](YoloModelFiles) folder, and can be used in prediction using the following code.
```python 
# The YOLO model is used to perform object detection on an image and draw the bounding box on the image
# Load an image
img = cv2.imread(image_path)

# Load a YOLO model
model = YOLO(model_path)

# Perform object detection on the image
results = model(image_path)[0]

# Get the bounding box coordinates
box = results.boxes.xyxy[0]
box_cpu = box.cpu()
box_int = box_cpu.numpy().astype(np.int32)

# Draw the bounding box on the image
cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (0, 255, 0), 2)

# Display the image
plt.imshow(img)
```
### 2. Corner detection using WoodenBox_detector Package

#### [WoodenBox_detector](WoodenBox_detector\WoodenBox_detector.py) Package documentation

- `run`: Runs the `WoodenBox` object to detect wooden boxes. It first calls the GetMask method to generate a list of binary masks using YOLO segmentation. If the detection is successful (i.e., masks exist), it calls the `GetMinBBox` method to get the minimum bounding box. Finally, it returns the minimum bounding box.

- `GetMask`: Uses YOLO segmentation to generate a list of binary masks. It initializes a YOLO model with the given weights path, performs YOLO segmentation on the input image with the specified confidence threshold, and stores the masks in the masks attribute if they exist.

- `GetCorners`: Returns the detected corners using Harris corner detection on a binary mask. It converts the binary mask to a float32 numpy array, applies Harris corner detection to the grayscale image, and returns the detected corners as a binary image.

- `GetMinBBox`: Returns the minimum (rotated) box corners using a binary mask. It finds the contours of the mask using external retrieval mode and simplified approximation, sorts the contours in descending order based on their area, finds the minimum area rectangle (minimum bounding box) for the largest contour, and returns the corners of the minimum bounding box.
#### Usage
```python
from WoodenBox_detector import WoodenBox

# Create a WoodenBox object
wooden_box = WoodenBox(image='path/to/image.jpg', weights_path='path/to/weights.pt')

# Run the detection
bounding_box = wooden_box.run()

# Check if the detection was successful
if bounding_box is not None:
    # Process the bounding box
    # ...

    # Display the result
    # ...
else:
    print("Wooden box not detected.")

```
### 3. 3D model generation
This [notebook demonstrates](depthToPointCloud.ipynb) how to convert RGBD (Grayscale + Depth) image to point cloud using Open3D library. The example is using Prime Sense camera intrinsic parameters. The notebook contains the following steps:

1. Load RGBD images from png files.
2. Convert RGBD image to point cloud using Open3D library.
3. Display the images and their depth maps.
4. Apply transformation to flip the point cloud upside down.
5. Visualize the point cloud using Open3D library.

