import cv2
from deepface import DeepFace

# Load face detection model
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract face region
        face = frame[y:y+h, x:x+w]

        try:
            # Emotion detection
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]

            text = f"{emotion} ({confidence:.1f}%)"

            # Display emotion above face
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        except:
            pass

    # Show output
    cv2.imshow("Emotion Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
