from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
from inference import get_model
import base64

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

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
        list_emotion = result['emotion']
    except Exception as e:
        emotion = None
        list_emotion = {}
    list_emotion = {k: float(v) for k, v in list_emotion.items()} if list_emotion else {}
    return jsonify({"emotion": emotion, "list_emotion": list_emotion})

if __name__ == "__main__":
    app.run(debug=True, port=3000)