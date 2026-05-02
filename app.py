import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
# Configuración agresiva de CORS para evitar bloqueos en el navegador
CORS(app, resources={r"/*": {"origins": "*"}})

# Cargar el modelo YOLO
# Asegúrate de que la ruta coincida con tu archivo .pt en Railway
model_path = 'runs/detect/yolo_control/weights/best.pt'
model = YOLO(model_path)

# Mapeo de nombres (opcional, para asegurar consistencia)
SMART_MAPPING = {
    "Atun": "Atun",
    "Deo Pies": "Deo Pies",
    "Maiz en lata": "Maiz en lata"
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "IA Service is UP", "model": model_path})

@app.route('/detect', methods=['POST'])
def detect_single():
    """Ruta antigua para detectar solo el objeto principal (compatibilidad)"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        results = model.predict(source=img, conf=0.6, save=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({"product": "unknown", "confidence": 0})

        box = results[0].boxes[0]
        detected_class = model.names[int(box.cls[0])]
        
        return jsonify({
            "product": SMART_MAPPING.get(detected_class, detected_class),
            "confidence": float(box.conf[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect_all', methods=['POST'])
def detect_all():
    """Nueva ruta para detectar TODOS los objetos en el estante"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Predecir con un umbral de confianza del 55% para capturar todo bien
        results = model.predict(source=img, conf=0.55, save=False)
        
        detected_products = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detected_class = model.names[int(box.cls[0])]
                name = SMART_MAPPING.get(detected_class, detected_class)
                detected_products.append(name)

        print(f"👁️ Estante detectado: {detected_products}")
        return jsonify({
            "products": detected_products,
            "count": len(detected_products)
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Railway usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)