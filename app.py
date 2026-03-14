from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app) # Permite que React (5173) le hable a Python (5000)

# Cargamos el modelo YOLOv8 (se descargará solo la primera vez)
model = YOLO('yolov8n.pt') 

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'image' not in request.files:
        return jsonify({"error": "No hay imagen"}), 400
    
    # Leer la imagen enviada por React
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # YOLO analiza la imagen
    results = model.predict(img)
    
    # Extraer el nombre del objeto detectado (ej: "apple")
    detectados = []
    for r in results:
        for c in r.boxes.cls:
            detectados.append(model.names[int(c)])

    # Devolvemos el primer objeto que encontró
    nombre_detectado = detectados[0] if detectados else "unknown"
    return jsonify({"producto": nombre_detectado})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    