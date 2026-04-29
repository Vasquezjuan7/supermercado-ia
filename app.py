from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
# Aggressive CORS configuration
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load the custom model trained locally with Roboflow annotations
model = YOLO('runs/detect/yolo_control/weights/best.pt')

# Smart mapping: maps YOLO class names to friendly inventory names
SMART_MAPPING = {
    "Atun": "Tuna",
    "Deo Pies": "Foot Deodorant",
    "Maiz en lata": "Canned Corn"
}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if an image was provided in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Run YOLO prediction at reduced resolution (320px) to save RAM on Render
        results = model.predict(img, conf=0.25, imgsz=320, verbose=False)
        
        # Extract detected class labels
        labels = [model.names[int(c)] for r in results for c in r.boxes.cls]
        
        # Free memory explicitly after prediction
        import gc
        gc.collect()
        
        if not labels:
            return jsonify({"product": "unknown"})

        # Map the first detection to a friendly name
        detected = labels[0]
        final_name = SMART_MAPPING.get(detected, detected.capitalize())

        return jsonify({"product": final_name})

    except Exception as e:
        # Return the actual error to the frontend for diagnosis
        error_msg = str(e)
        print(f"🔥 ERROR CRÍTICO IA: {error_msg}")
        return jsonify({
            "error": "AI Server error",
            "details": error_msg
        }), 500

if __name__ == '__main__':
    # Use port 7860 by default (Hugging Face Spaces standard)
    port = int(os.environ.get("PORT", 7860))
    # host='0.0.0.0' required so the container accepts external connections
    app.run(host='0.0.0.0', port=port, debug=False)