from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, session
from openai import OpenAI
from deepface import DeepFace
import cv2
import numpy as np
import mediapipe as mp
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
import math

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    """Display live emotion detection with embedded chatbot."""
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

    # Compose the prompt for GPT
    prompt = (
        f"Subject is feeling {emotion}. "
        f"Here are the emotion probabilities: {emotions}. "
        "Explain in 2-3 sentences why the subject might be feeling this way."
    )

    # Call OpenAI API for explanation
    try:
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history,
        )
        explanation = completion.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"Error: {e}"

    return jsonify({
        "emotion": emotion,
        "emotions": emotions,
        "prompt": prompt,
        "explanation": explanation
    })

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
        interval = int(fps * 5)
        emotions_timeline = []
        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        processed_filename = "processed_" + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

        # First pass: DeepFace emotion timeline
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

        # Eye landmarks and blink detection
        LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
        RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]
        IRIS_LEFT = [468, 469, 470, 471, 472]
        IRIS_RIGHT = [473, 474, 475, 476, 477]
        BLINK_THRESHOLD = 0.20  
        blink_times = []
        prev_eye_open = True
        frame_number = 0

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            cap = cv2.VideoCapture(filepath)
            out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw eye landmarks
                        for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + IRIS_LEFT + IRIS_RIGHT:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * frame.shape[1])
                            y = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        left_top = face_landmarks.landmark[159]
                        left_bottom = face_landmarks.landmark[145]
                        left_left = face_landmarks.landmark[33]
                        left_right = face_landmarks.landmark[133]

                        vert_dist = euclidean((left_top.x, left_top.y), (left_bottom.x, left_bottom.y))
                        horiz_dist = euclidean((left_left.x, left_left.y), (left_right.x, left_right.y))
                        ear = vert_dist / horiz_dist if horiz_dist != 0 else 0

                        if ear < BLINK_THRESHOLD and prev_eye_open:
                            blink_time = frame_number / fps
                            blink_times.append(blink_time)
                            prev_eye_open = False
                            cv2.putText(frame, "BLINK!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                            emotions_timeline.append({"time": round(blink_time, 2), "emotion": "Blink"})
                        elif ear >= BLINK_THRESHOLD:
                            prev_eye_open = True

                out.write(frame)
                frame_number += 1
            cap.release()
            out.release()

        return render_template(
            "video_result.html",
            video_url=url_for('uploaded_file', filename=processed_filename),
            emotions=emotions_timeline,
            filename=filename
        )

    # GET request: show upload form
    return render_template("video_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/abuse_detect", methods=["GET"])
def abuse_detect():
    return render_template("abuse_detect.html")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False, mimetype='video/mp4')

@app.route("/chatbot", methods=["GET"])
def chatbot_page():
    """Display a very small chatbot interface."""
    return render_template("chatbot.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Generate a dynamic response using the OpenAI API."""
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please say something."})

    # Maintain conversation history in the session
    history = session.get("chat_history", [
        {"role": "system", "content": "You are a helpful assistant."}
    ])
    history.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history,
        )
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        response_text = f"Error: {e}"

    history.append({"role": "assistant", "content": response_text})
    session["chat_history"] = history[-6:]  # keep last few messages





    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
