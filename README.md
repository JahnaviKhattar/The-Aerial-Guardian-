# Aerial Guardian: Drone Based Multi Object Person Tracking System

## Overview

Aerial Guardian is a lightweight computer vision pipeline designed for detecting and tracking multiple persons from moving drone footage.

The system focuses on handling challenges present in aerial videos:

* small object sizes
* high camera movement
* object occlusion
* identity consistency during tracking

The pipeline combines YOLO based object detection with ByteTrack multi-object tracking and additional drone-specific optimizations.

## Architecture

Input Drone Video

↓

Frame Stabilization (ORB Feature Matching)

↓

Tiled Frame Processing for Small Object Detection

↓

YOLOv8 Person Detector

↓

ByteTrack Multi Object Tracker

↓

Trajectory Visualization + FPS Monitoring

↓

Processed Output Video

## Model Choice

YOLOv8 Nano was selected as the detection backbone because it provides a good balance between inference speed and detection accuracy.

Reasons:

* lightweight model size
* suitable for edge devices
* fast inference
* good real-time capability

## Small Object Detection Improvements

Drone videos contain very small person instances. To improve detection:

1. Higher resolution inference was used.
2. Frames were divided into smaller tiles before detection.
3. Confidence thresholds were adjusted for aerial scenarios.

This improves visibility of small objects without switching to a heavier model.

## Tracking Approach

ByteTrack was used for Multi Object Tracking.

The tracker maintains consistent identities by associating detections between consecutive frames.

Features:

* unique person IDs
* persistent tracking
* trajectory history visualization

## Handling Drone Ego Motion

Moving drones introduce camera motion which may cause ID switching.

To reduce this:

* ORB features are extracted between consecutive frames.
* Feature matching estimates camera movement.
* Affine transformation is used for stabilization before tracking.

This improves tracking consistency.

## Optimizations

The system was designed considering edge deployment.

Optimization choices:

* YOLOv8 nano model
* lightweight tracking algorithm
* optimized preprocessing
* no heavy GPU dependency

## Edge Deployment (NVIDIA Jetson)

For deployment on Jetson hardware:

* Convert YOLO model to ONNX
* Use TensorRT acceleration
* Enable FP16 precision
* Reduce unnecessary computations
* Optimize input resolution based on FPS requirement

## Output

The final output video contains:

* Person bounding boxes
* Unique tracking IDs
* Movement trajectory tails
* FPS information
* Person analytics

<img width="923" height="652" alt="output" src="https://github.com/user-attachments/assets/b2880dfe-4cfe-43d7-ba98-e0d2a9afbc01" />

## Technologies Used

* Python
* OpenCV
* YOLOv8
* ByteTrack
* NumPy
* Supervision

## Running Instructions

Install dependencies:

pip install -r requirements.txt

Run:

python main.py
