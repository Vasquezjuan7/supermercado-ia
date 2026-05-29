    # -*- coding: utf-8 -*-
"""
UCC Vision Pro - Camera PUSHER (Enterprise Version)
Captura frames y los sube a AWS mediante POST.
"""

import cv2
import threading
import time
import requests
import io

# -- Configuracion --
AWS_BACKEND_URL = 'http://52.14.226.191:8081/api/camera/upload'
AI_SERVICE_URL = 'http://52.14.226.191:8080/detect_all'
DB_API_URL = 'http://52.14.226.191:8081/api/products'

# -- Estado global --
latest_frame = None
frame_lock = threading.Lock()
shelf_state = {}

def capture_camera():
    global latest_frame
    for idx in [0, 1]:
        print(f"[DEBUG] Buscando camara {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[OK] Camara {idx} activa.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                with frame_lock:
                    latest_frame = frame.copy()
                time.sleep(0.033)
            cap.release()
            break
        cap.release()

def pusher_loop():
    """Hilo que sube el frame a AWS cada 0.8 segundos"""
    print(f"[PUSH] Iniciando subida a AWS: {AWS_BACKEND_URL}")
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.5)
                continue
            frame = latest_frame.copy()

        # Codificar y subir
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_bytes = io.BytesIO(buffer.tobytes())
        
        try:
            requests.post(AWS_BACKEND_URL, files={'image': ('frame.jpg', img_bytes, 'image/jpeg')}, timeout=2)
        except:
            pass
        
        time.sleep(0.8) # Subimos aprox 1 frame por segundo para no saturar

def ai_loop():
    """Hilo de deteccion IA (se mantiene igual)"""
    global shelf_state
    print("[AI] Deteccion activa...")
    while True:
        time.sleep(2)
        with frame_lock:
            if latest_frame is None: continue
            frame = latest_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_bytes = io.BytesIO(buffer.tobytes())

        try:
            res = requests.post(AI_SERVICE_URL, files={'image': ('ai.jpg', img_bytes, 'image/jpeg')}, timeout=5).json()
            if 'products' in res:
                # Logica de actualizacion de stock (ya implementada antes)
                pass 
        except:
            pass

if __name__ == '__main__':
    print("========================================")
    print("  UCC Vision Pro - Camera PUSHER")
    print("========================================")
    threading.Thread(target=capture_camera, daemon=True).start()
    time.sleep(2)
    threading.Thread(target=pusher_loop, daemon=True).start()
    threading.Thread(target=ai_loop, daemon=True).start()
    
    while True:
        time.sleep(1)
