from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

# Cargamos el modelo Nano (pesa solo 6MB, ideal para Render)
model = YOLO('yolov8n.pt') 

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'image' not in request.files:
        return jsonify({"error": "No hay imagen"}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Optimizamos la imagen para que Render no se quede sin RAM
    img_resized = cv2.resize(img, (320, 320))
    results = model.predict(img_resized, conf=0.25)
    
    detectados = []
    for r in results:
        for c in r.boxes.cls:
            detectados.append(model.names[int(c)])

    nombre_detectado = detectados[0] if detectados else "unknown"
    return jsonify({"producto": nombre_detectado})

if __name__ == '__main__':
    # Puerto dinámico para Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
