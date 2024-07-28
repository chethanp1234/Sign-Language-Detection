import os
from flask import Flask, request, jsonify, send_from_directory
import subprocess
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

# Paths to the model and labels
model_path = "Sign_Language_Detection/Model/keras_model.h5"
labels_path = "Sign_Language_Detection/Model/labels.txt"
labels = ["Hello", "ILoveYou", "Thankyou", "Yes"]

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/run_test', methods=['POST'])
def run_test():
    subprocess.Popen(["python", "Sign_Language_Detection/test.py"])
    return '', 204

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = img[y-20:y + h + 20, x-20:x + w + 20]

        if imgCrop.size != 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            result = labels[index]
        else:
            result = 'No hand detected'
    else:
        result = 'No hand detected'

    os.remove(filepath)
    return jsonify({'result': [result]})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    frame_count = 0
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 15 == 0:  # Sample every 15 frames
            hands, frame = detector.findHands(frame)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                imgCrop = frame[y-20:y + h + 20, x-20:x + w + 20]

                if imgCrop.size != 0:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = 300 / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, 300))
                        wGap = math.ceil((300 - wCal) / 2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                    else:
                        k = 300 / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (300, hCal))
                        hGap = math.ceil((300 - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    detections.append(labels[index])

    cap.release()
    os.remove(filepath)

    # Reduce the number of outputs by taking unique values
    unique_detections = list(set(detections))
    return jsonify({'result': unique_detections})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
