import cv2
from ultralytics import YOLO

def main():
    print("[*] Loading generic factory model (yolo11n.pt)...")
    # Cargar el modelo GENERICO de YOLO11
    model = YOLO('yolo11n.pt')

    # Abrir la cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n==================================")
    print(" 🎥 GENERIC AI DETECTION STARTED")
    print("==================================")
    print("-> Point the camera around your room!")
    print("-> Press 'Q' on your keyboard to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            break

        # La IA analiza el video en tiempo real (conf=0.5 significa 50% de confianza)
        results = model.predict(frame, conf=0.5, verbose=False)

        # YOLO dibuja las cajas automáticamente
        annotated_frame = results[0].plot()

        # Mostramos la ventana
        cv2.imshow("UCC Market - Generic AI", annotated_frame)

        # Salir con la tecla Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
