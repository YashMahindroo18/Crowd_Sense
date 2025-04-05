def func(video, mask):
    import numpy as np
    from ultralytics import YOLO
    import cv2
    import cvzone
    import math
    from sort import Sort

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0, 0

    model = YOLO("../Yolo_Weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    mask = cv2.imread(mask)
    if mask is None:
        print("Error: Could not read mask image.")
        return 0, 0
    mask = cv2.resize(mask, (1280, 720))

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limitsUp = [103, 161, 296, 161]
    limitsDown = [527, 489, 735, 489]

    totalCountUp = []
    totalCountDown = []

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("End of video or failed to read frame.")
            break

        imgregion = cv2.bitwise_and(img, mask)
        imgGraphics = cv2.imread(r"D:\minor_project\graphics-1.png", cv2.IMREAD_UNCHANGED)
        if imgGraphics is not None:
            img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

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

        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
                if id not in totalCountUp:
                    totalCountUp.append(id)
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                if id not in totalCountDown:
                    totalCountDown.append(id)
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

        #cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return len(totalCountUp), len(totalCountDown)


from flask import Flask, render_template, request, jsonify
# from People_counter.My_version import func
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/count', methods=['POST'])
def count_people():
    if 'video' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    video = request.files['video']
    mask = request.files['mask']
    
    # Save uploaded files
    os.makedirs('static/uploads', exist_ok=True)

    video_path = os.path.join('static/uploads', video.filename)
    mask_path = os.path.join('static/uploads', mask.filename)
    video.save(video_path)
    mask.save(mask_path)

    # Call the counting function
    up, down = func(video_path, mask_path)
    
    return jsonify({'up': up, 'down': down})

if __name__ == '__main__':
    app.run(debug=True)

