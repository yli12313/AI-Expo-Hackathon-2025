from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, Response
from deepface import DeepFace
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
SURVEILLANCE_FOLDER = "surveillance"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SURVEILLANCE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SURVEILLANCE_FOLDER'] = SURVEILLANCE_FOLDER

# Initialize YOLO model
yolo_model = YOLO("yolov8n.pt")

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def draw_yolo_boxes(frame, yolo_results, yolo_model):
    """Draw YOLO detection boxes on frame"""
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
    """Generate video frames with YOLO detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop the video for preview
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Apply YOLO detection
            yolo_results = yolo_model(frame)
            frame_with_boxes = draw_yolo_boxes(frame, yolo_results, yolo_model)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def get_surveillance_videos():
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    return render_template("live_result.html")

@app.route("/playback")
def playback():
    videos = get_surveillance_videos()
    return render_template("playback.html", videos=videos)

@app.route("/video_analysis/<filename>")
def video_analysis(filename):
    """Route for individual video analysis page"""
    # Verify the file exists in surveillance folder
    video_path = os.path.join(SURVEILLANCE_FOLDER, filename)
    if not os.path.exists(video_path) or not allowed_video_file(filename):
        return "Video not found", 404
    
    video_info = {
        'filename': filename,
        'name': os.path.splitext(filename)[0],
        'path': video_path,
        'size': os.path.getsize(video_path)
    }
    return render_template("video_analysis.html", video=video_info)

@app.route("/surveillance/<filename>")
def serve_surveillance_video(filename):
    """Serve video files from surveillance folder"""
    return send_from_directory(SURVEILLANCE_FOLDER, filename)

@app.route("/surveillance_yolo/<filename>")
def serve_yolo_video_stream(filename):
    """Serve video stream with YOLO detection for preview"""
    video_path = os.path.join(SURVEILLANCE_FOLDER, filename)
    if not os.path.exists(video_path) or not allowed_video_file(filename):
        return "Video not found", 404
    
    return Response(generate_yolo_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    file = request.files['frame']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        
        emotion = result['dominant_emotion']
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"emotion": "No face detected", "error": str(e)})

@app.route("/analyze_video", methods=["GET", "POST"])
def analyze_video():
    if request.method == "POST":
        file = request.files['video']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * 5)
        emotions_timeline = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    emotion = result['dominant_emotion']
                except Exception:
                    emotion = "Unknown"
                emotions_timeline.append({"time": round(frame_idx / fps, 2), "emotion": emotion})
            frame_idx += 1
        cap.release()

        return render_template("video_result.html", video_url=url_for('uploaded_file', filename=filename), emotions=emotions_timeline)

    return render_template("video_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=3000)