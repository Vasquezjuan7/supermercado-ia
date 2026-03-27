from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

# Load the nano model (lightweight for Render)
model = YOLO('yolov8n.pt') 

# DICCIONARIO DE TRADUCCIÓN Y CATEGORIZACIÓN
# Aquí mapeamos lo que YOLO detecta a lo que tu supermercado necesita
SMART_MAPPING = {
    "bottle": "Soda / Water",
    "cup": "Milk Carton",
    "apple": "Fresh Apple",
    "orange": "Fresh Orange",
    "banana": "Banana",
    "broccoli": "Vegetables",
    "box": "Snacks Box",
    "handbag": "Shopping Bag",
    "person": "Customer (Wait)",
    "cell phone": "Digital Payment"
}

@app.route('/detect', methods=['POST']) # Cambiado a /detect
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Resize to 320x320 to avoid Render's Out of Memory (Status 137)
        img = cv2.resize(img, (320, 320))
        
        results = model.predict(img, conf=0.25)
        
        # Obtenemos la etiqueta original de YOLO
        raw_labels = [model.names[int(c)] for r in results for c in r.boxes.cls]
        
        if not raw_labels:
            return jsonify({"product": "unknown"})

        # Aplicamos la traducción/mapeo al primer objeto detectado
        detected_raw = raw_labels[0]
        translated_product = SMART_MAPPING.get(detected_raw, detected_raw.capitalize())
        
        return jsonify({"product": translated_product})

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
