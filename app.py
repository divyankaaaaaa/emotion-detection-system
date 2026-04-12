from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace
from flask import request
import numpy as np
current_emotion = "Waiting..."

app = Flask(__name__)

# Load face detection model
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(
                    face, actions=['emotion'], enforce_detection=False
                )

                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]

                text = f"{emotion} ({confidence:.1f}%)"
                global current_emotion
                current_emotion = text

                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

            except:
                pass

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Send frame to browser
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/emotion')
def emotion():
    return jsonify({"emotion": current_emotion})
    
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    
    if not file:
        return "No file uploaded"

    # Convert image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        return f"Emotion: {emotion} ({confidence:.1f}%)"

    except:
        return "Error detecting emotion"


if __name__ == "__main__":
    app.run(debug=True)