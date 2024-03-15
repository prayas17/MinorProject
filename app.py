import tensorflow as tf
import numpy as np
import cv2
import pickle
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load model and LabelEncoder
model = tf.keras.models.load_model("FacialExpressionModel.h5")
Le = pickle.load(open("LabelEncoder.pck", "rb"))

# Function to process image
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [96, 96], method="bilinear")
    image = tf.expand_dims(image, 0)
    return image

# Function for real-time prediction
def realtime_prediction(image, model, encoder_):
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=1)
    return encoder_.inverse_transform(prediction)[0]

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and perform emotion recognition
def detect_emotions():
    VideoCapture = cv2.VideoCapture(0)
    while True:
        ret, frame = VideoCapture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) >= 1:
            for (x, y, w, h) in faces:
                img = gray[y-10: y+h+10, x-10: x+w+10]
                if img.shape[0] == 0 or img.shape[1] == 0:
                    cv2.imshow("Frame", frame)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = process_image(img)
                    out = realtime_prediction(img, model, Le)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    z = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(frame, str(out), (x, z), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    # Display the detected emotion separately
                    cv2.putText(frame, "Emotion: " + str(out), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    VideoCapture.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotions(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
