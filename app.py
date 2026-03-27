from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

model = YOLO('yolov8n.pt') 

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'image' not in request.files:
        return jsonify({"error": "No hay imagen"}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=0.25)
    
    detectados = []
    for r in results:
        for c in r.boxes.cls:
            detectados.append(model.names[int(c)])

    nombre_detectado = detectados[0] if detectados else "unknown"
    return jsonify({"producto": nombre_detectado})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)