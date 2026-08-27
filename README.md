# Crowd Sense - Real-time Crowd Detection & Tracking

## Overview
A computer vision application that detects and tracks people in video streams using YOLO and the SORT algorithm.

## Features
- Real-time object detection using YOLOv8
- Multi-object tracking with SORT algorithm
- Process video files and webcam streams
- Annotated output with bounding boxes and track IDs

## Tech Stack
- **Python** - Core application
- **YOLOv8** - Object detection
- **OpenCV** - Video processing
- **SORT** - Object tracking algorithm

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/YashMahindroo18/Crowd_Sense.git
cd Crowd_Sense
pip install -r requirements.txt
```

### Usage
```bash
python app.py
```

Upload a video file or connect a webcam stream to detect and track people in real-time.

## How It Works
1. YOLO detects objects (people) in each frame
2. SORT algorithm maintains consistent tracking IDs across frames
3. Annotated video with bounding boxes and IDs is saved/displayed

## Future Improvements
- Web UI for easy uploading and visualization
- REST API backend for integration
- Crowd density heatmaps
- Deploy on cloud platform

