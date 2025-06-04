### Presentation is [here](https://docs.google.com/presentation/d/1z-OlNjtZoe0iGm4j799QXGNu3lbD7xMHCoyMln45Xsg/edit). We tried really hard and built a cool project.
* **Team**: Joshua John, Yue (Michelle) Lei, Yingquan Li

# Project Perception

![image](https://github.com/user-attachments/assets/5e96550c-5619-411d-a931-d93a1c48673a)

Project Perception is a video analysis platform designed for interrogation and surveillance scenarios. It uses advanced computer vision and AI to detect emotions, blinks, and objects in both live and uploaded video footage. The platform also features a chatbot assistant for querying results and explanations.

---

## Features

- **Live Emotion Detection:** Analyze emotions in real-time using your webcam.
- **Video Upload & Analysis:** Upload videos for automated emotion, blink, and object detection.
- **YOLOv8 Object Recognition:** Detects and labels objects in video frames.
- **Playback & History:** Review previously analyzed videos and their results.
- **Chatbot Assistant:** Ask questions about analysis results or get explanations.

---

## Tech Stack

- **Backend:** Python 3.12, Flask, OpenCV, DeepFace, YOLOv8 (ultralytics), Torch, Pandas, NumPy, OpenAI API
- **Frontend:** HTML5, CSS3 (Bebas Neue, Inter), Vanilla JavaScript, Font Awesome
- **Storage:** Local file storage for uploads and processed results

---

## Project Structure

```
app.py
requirements.txt
dev-requirements.txt
utils.py
yolov8n.pt
static/           # CSS, JS, images
templates/        # HTML templates
surveillance/     # Sample videos
uploads/          # User uploads
```

---

## Getting Started

1. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the application:**
    ```sh
    python app.py
    ```

3. **Access the app:**  
   Open [http://localhost:5000](http://localhost:5000) in your browser.


4. **Chatbot**
   To get the chatbot working, put your OPEN AI API key in a `.env` file:
   ```sh
   OPENAI_API_KEY=<YOUR_SECRET_KEY>
   ```

---

## Usage

- Use the dashboard to start live analysis or upload a video.
- View results with detected emotions, blinks, and objects.
- Access playback/history to review past analyses.
- Interact with the chatbot for insights or explanations.

---

## License

For demonstration and educational purposes only.
