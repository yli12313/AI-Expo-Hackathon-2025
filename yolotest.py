import cv2

from ultralytics import solutions

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