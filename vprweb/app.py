from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from VPRmodel import VPRmodel

app = Flask(__name__)

def resize_image(image, target_size=(640, 480)):
    h, w = image.shape[:2]
    scale = max(target_size[0] / w, target_size[1] / h)

    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h, w = resized_image.shape[:2]
    startx = w//2 - target_size[0]//2
    starty = h//2 - target_size[1]//2

    return resized_image[starty:starty+target_size[1], startx:startx+target_size[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image was uploaded"}), 400

    filestr = request.files['image'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = resize_image(cv2.imdecode(npimg, cv2.IMREAD_COLOR))

    return jsonify({"message": vpr.predict(img)})

if __name__ == '__main__':
    vpr = VPRmodel(True, 10, 'yolov5x')
    app.run(debug=True)
