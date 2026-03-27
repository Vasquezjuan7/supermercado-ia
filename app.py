from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
# Permitimos CORS para que Vercel pueda conectar sin problemas
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. Cargamos el modelo (se descargará automáticamente)
model = YOLO('yolov8n.pt')

# 2. Tu diccionario de supermercado UCC
SMART_MAPPING = {
    "apple": "Fresh Apple",
    "orange": "Orange",
    "banana": "Banana",
    "bottle": "Soda / Water",
    "cup": "Milk Box",
    "box": "Snacks Box",
    "sandwich": "Prepared Food"
}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Verificamos si hay imagen
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # 3. Predicción con YOLO (Ahora con resolución completaimgs)
        # imgsz=640 es el estándar y tenemos RAM para procesarlo
        results = model.predict(img, conf=0.25, imgsz=640)
        
        # Obtenemos las etiquetas de YOLO
        labels = [model.names[int(c)] for r in results for c in r.boxes.cls]
        
        if not labels:
            return jsonify({"product": "unknown"})

        # Mapeamos la primera detección
        detected = labels[0]
        final_name = SMART_MAPPING.get(detected, detected.capitalize())

        return jsonify({"product": final_name})

    except Exception as e:
        # Registramos el error en los logs de Hugging Face
        print(f"ERROR: {str(e)}")
        return jsonify({"error": "IA Server error"}), 500

if __name__ == '__main__':
    # Hugging Face usa el puerto 7860 por defecto
    port = int(os.environ.get("PORT", 7860))
    # Importante: host='0.0.0.0' para que el contenedor escuche
    app.run(host='0.0.0.0', port=port, debug=False)
