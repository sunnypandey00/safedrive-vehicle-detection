# SafeDrive: Real-Time Vehicle Detection for Edge Devices

## Problem Statement
Developing a robust, real-time vehicle detection system for deployment on constrained edge devices like dashcams. Edge hardware possesses severe computational and memory limitations, requiring an object detection model that is exceptionally lightweight yet highly accurate. The system must consistently detect diverse traffic elements (cars, trucks, bikes, pedestrians) at high frame rates (30+ FPS) to ensure immediate response times for driver safety and traffic analysis applications.

## Project Overview
This repository contains a complete, end-to-end computer vision pipeline designed to solve the edge-inference challenge. The project focuses on high-speed inference and lightweight architecture, featuring both state-of-the-art fine-tuned models (YOLOv8) and a custom PyTorch YOLOv1 architecture built entirely from scratch.

## Key Features
* **Advanced Data Augmentation:** Implemented a robust `Albumentations` pipeline including horizontal flipping, random brightness/contrast, and motion blur to simulate real-world dashcam conditions.
* **YOLOv8 Fine-Tuning & Optimization:** Trained and compared YOLOv8 Nano and Small variants to find the optimal balance between mean Average Precision (mAP) and inference speed.
* **Custom YOLOv1 PyTorch Architecture:** Designed and trained a lightweight, 6-layer custom Convolutional Neural Network (CNN) mirroring the YOLOv1 grid-based detection system. 
* **Custom Composite Loss Function:** Implemented a ground-up YOLO loss function handling coordinate loss, objectness, no-objectness, and classification loss using PyTorch.
* **Edge Deployment Ready:** Successfully exported the deployment model to ONNX format for hardware-agnostic, accelerated edge inference.
* **High-Speed Video Inference:** Achieved real-time video processing at 60+ FPS on traffic dashcam footage.

## Model Comparison & Selection

Both YOLOv8 Nano and Small models were trained on the augmented dataset to evaluate the best fit for edge hardware.

| Model Variant | Parameters | Inference Speed | mAP@50 |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Nano** | 3.0 M | 2.4 ms (~60 FPS) | 0.898 |
| **YOLOv8 Small** | 11.1 M | 4.8 ms (~30 FPS) | 0.929 |

**Conclusion:** The YOLOv8 Nano model was selected as the final deployment model. While the Small variant offered a marginal 3% increase in precision, the Nano model delivered twice the inference speed and required significantly less memory. At over 60 FPS, the Nano model prevents frame dropping on constrained edge devices.

## Repository Structure

* `safedrive_vehicle_detection.ipynb`: The main Jupyter Notebook containing the entire pipeline, including data preparation, Ultralytics training, PyTorch custom modeling, and inference scripts.
* `yolov8n_safedrive.pt`: The fine-tuned PyTorch weights for the chosen YOLOv8 Nano model.
* `yolov8s_safedrive.pt`: The fine-tuned PyTorch weights for the YOLOv8 Small model (used for baseline comparison).
* `yolov8n_safedrive.onnx`: The ONNX-exported version of the Nano model, optimized for edge deployment.
* `yolov1_scratch_custom.pth`: The trained state dictionary for the custom from-scratch PyTorch YOLOv1 architecture.
* `yolov8n_inference_grid.png`: A 9-image grid demonstrating the Nano model's bounding box accuracy on the validation set.
* `dashcam_inference_demo.mp4`: A sample dashcam video processed by the pipeline, demonstrating real-time vehicle tracking and classification.

## Technical Stack
* **Frameworks:** PyTorch, Ultralytics (YOLOv8)
* **Computer Vision:** OpenCV, Albumentations
* **Deployment:** ONNX Runtime
* **Data Visualization:** Matplotlib, Seaborn