from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os  # Necesario para leer el puerto de Railway

app = Flask(__name__)
# Permitimos CORS para que tu link de Vercel pueda hablar con este servicio
CORS(app) 

# Cargamos el modelo YOLOv8 Nano (es el más ligero y rápido para la nube)
model = YOLO('yolov8n.pt') 

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'image' not in request.files:
        return jsonify({"error": "No hay imagen"}), 400
    
    # 1. Leer la imagen que envía el Frontend (Vercel)
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # 2. Ejecutar la IA (YOLO)
    # conf=0.25 ayuda a que sea más preciso filtrando ruidos
    results = model.predict(img, conf=0.25)
    
    # 3. Extraer nombres de lo detectado
    detectados = []
    for r in results:
        for c in r.boxes.cls:
            detectados.append(model.names[int(c)])

    # 4. Responder con el primer objeto o "unknown"
    nombre_detectado = detectados[0] if detectados else "unknown"
    
    print(f"IA detectó: {nombre_detectado}") # Esto lo verás en los logs de Railway
    return jsonify({"producto": nombre_detectado})

# --- CONFIGURACIÓN PARA RAILWAY/NUBE ---
if __name__ == '__main__':
    # Railway inyecta la variable PORT automáticamente
    port = int(os.environ.get("PORT", 5000))
    # Importante: host='0.0.0.0' para que sea accesible externamente
    app.run(host='0.0.0.0', port=port)