import os
import math
import cv2
import numpy as np
from ultralytics import YOLO

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

yolo_model = YOLO("yolov8n.pt")
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def draw_yolo_boxes(frame, yolo_results, yolo_model):
    for result in yolo_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (90, 255, 90)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
    return frame

def generate_yolo_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yolo_results = yolo_model(frame)
            frame_with_boxes = draw_yolo_boxes(frame, yolo_results, yolo_model)
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def get_surveillance_videos(SURVEILLANCE_FOLDER):
    """Get list of video files in surveillance folder"""
    videos = []
    if os.path.exists(SURVEILLANCE_FOLDER):
        for filename in os.listdir(SURVEILLANCE_FOLDER):
            if allowed_video_file(filename):
                videos.append({
                    'filename': filename,
                    'name': os.path.splitext(filename)[0],
                    'path': os.path.join(SURVEILLANCE_FOLDER, filename),
                    'size': os.path.getsize(os.path.join(SURVEILLANCE_FOLDER, filename))
                })
    return sorted(videos, key=lambda x: x['name'])

def detect_movement(video_path, threshold=10000000, min_events=6):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    movement_events = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            movement_score = np.sum(thresh)
            if movement_score > threshold:
                movement_events += 1
                if movement_events >= min_events:
                    cap.release()
                    return True
        prev_frame = gray
    cap.release()
    return False
