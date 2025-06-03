from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from deepface import DeepFace
import cv2
import numpy as np
import mediapipe as mp
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    return render_template("live_result.html")

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
        if emotion in ["neutral", "disgust"]:
            emotion = "other"
        emotions = result['emotion']

    except Exception as e:
        emotion = None
        emotions = {}
    emotions = {k: float(v) for k, v in emotions.items()} if emotions else {}
    return jsonify({"emotion": "Subject is feeling: " + emotion, "emotions": emotions})

@app.route("/analyze_video", methods=["GET", "POST"])
def analyze_video():
    if request.method == "POST":
        file = request.files['video']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            return "<h1>Error: Could not open video file</h1>"

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = int(fps * 10)
        emotions_timeline = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                    if isinstance(result, list) and result:
                        emotion = result[0]['dominant_emotion']
                    else:
                        emotion = "Unknown"
                except Exception as e:
                    print(f"DeepFace analysis failed: {e}")
                    emotion = "Unknown"

                emotions_timeline.append({"time": round(frame_idx / fps, 2), "emotion": emotion})
            frame_idx += 1
        cap.release()

        return render_template(
            "video_result.html",
            video_url=url_for('uploaded_file', filename=filename),
            emotions=emotions_timeline
        )

    return render_template("video_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/abuse_detect", methods=["GET"])
def abuse_detect():
    return render_template("abuse_detect.html")

@app.route("/chatbot", methods=["GET"])
def chatbot_page():
    """Display a very small chatbot interface."""
    return render_template("chatbot.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Return a simple response for a given chat message."""
    data = request.get_json()
    message = data.get("message", "").strip().lower()

    canned = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help you?",
        "bye": "Goodbye!"
    }

    response = canned.get(message, f"You said: {message}")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

