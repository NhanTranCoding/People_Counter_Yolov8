from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import numpy as np

model = YOLO("../YoloWeights/yolov8n.pt")
classNames = ["person", 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
              'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
              'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

cap = cv2.VideoCapture("Videos/people.mp4")
cap.set(3, 1280)
cap.set(4, 720)
total_person_up = []
total_person_down = []
limits = [130, 200, 350, 200]
limits1 = [550, 450, 750, 450]
mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1280, 720))
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (700, 50))
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentClass = classNames[cls]
            if currentClass == "person":
                currentArray = np.array([x1, y1, x1 + w, y1 + h, conf])
                detections = np.vstack((detections, currentArray))
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                # cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (x1, y1), scale=0.8, thickness=1)
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w = x2 - x1
        h = y2 - y1
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
        cvzone.putTextRect(img, f"{id}", (x1, y1), scale=0.8, thickness=1)
        if limits[0] < cx < limits[2] and limits[1] -15 < cy < limits[1] + 15:
            if total_person_up.count(id) ==0:
                total_person_up.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        if limits1[0] < cx < limits1[2] and limits1[1] -15 < cy < limits1[1] + 15:
            if total_person_down.count(id) ==0:
                total_person_down.append(id)
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)

    cv2.putText(img, f"{len(total_person_up)}", (900, 130), cv2.FONT_HERSHEY_PLAIN, 4, (50, 50, 255), 3)
    cv2.putText(img, f"{len(total_person_down)}", (1150, 130), cv2.FONT_HERSHEY_PLAIN, 4, (50, 50, 255), 3)
    cv2.imshow("people", img)
    cv2.waitKey(1)