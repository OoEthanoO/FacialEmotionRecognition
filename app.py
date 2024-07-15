from flask import Flask, request, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
model = load_model('Emoti0.1.h5')

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    if request.method == 'POST':
        print("Received image")
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image_path = 'static/image.png'
        image.save(image_path)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

            roi = image_utils.img_to_array(roi_gray)
            roi = np.expand_dims(roi, axis=0)
            roi /= 255

            predictions = model.predict(roi)[0]
            label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion = label[np.argmax(predictions)]

            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imwrite(image_path, img)
        
        return send_file(image_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(port=3000, debug=True)
