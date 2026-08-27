import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from sort import Sort
import math

st.set_page_config(page_title="CrowdSense", layout="wide")

st.title("🎥 CrowdSense - Real-time Pedestrian Counter")
st.markdown("Upload a video and mask to count pedestrians crossing detection lines")

# Load model once (cache it)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# File uploads
col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
with col2:
    mask_file = st.file_uploader("Upload Mask", type=["png", "jpg", "jpeg"])

if video_file and mask_file:
    # Save uploaded files to temp
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        mask_path = os.path.join(tmpdir, "mask.png")
        
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        with open(mask_path, "wb") as f:
            f.write(mask_file.read())
        
        # Processing
        st.info("Processing video... This may take a moment")
        progress_bar = st.progress(0)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (1280, 720))
        
        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]
        
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        limitsUp = [103, 161, 296, 161]
        limitsDown = [527, 489, 735, 489]
        
        totalCountUp = []
        totalCountDown = []
        frame_count = 0
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            imgregion = cv2.bitwise_and(img, mask)
            results = model(imgregion, stream=True)
            detections = np.empty((0, 5))
            
            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0].item() * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    
                    if currentClass == "person" and conf > 0.3:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))
            
            resultsTracker = tracker.update(detections)
            
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                
                if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
                    if id not in totalCountUp:
                        totalCountUp.append(id)
                
                if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                    if id not in totalCountDown:
                        totalCountDown.append(id)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        
        # Display results
        st.success("✅ Processing complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pedestrians Going Up", len(totalCountUp), delta=len(totalCountUp))
        with col2:
            st.metric("Pedestrians Going Down", len(totalCountDown), delta=len(totalCountDown))
        
        st.markdown(f"**Total Unique Pedestrians Detected:** {len(totalCountUp) + len(totalCountDown)}")
