from flask import Flask, request, jsonify
from flask_cors import CORS # IMPORTANTE
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) # ESTA LÍNEA ES VITAL PARA VERCEL

# Usamos el modelo nano y forzamos que no use archivos pesados
model = YOLO('yolov8n.pt') 

@app.route('/detectar', methods=['POST'])
def detectar():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No hay imagen"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # REDIMENSIONAR es obligatorio para que Render no se apague (Status 137)
        img = cv2.resize(img, (320, 320))
        
        results = model.predict(img, conf=0.25)
        detectados = [model.names[int(c)] for r in results for c in r.boxes.cls]
        
        return jsonify({"producto": detectados[0] if detectados else "unknown"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
