import cv2
from ultralytics import YOLO
import time

# =============================================
# Inventory catalog - matches your trained classes
# =============================================
INVENTORY = {
    "Atun":        {"aisle": "Aisle 1", "emoji": "🐟"},
    "Deo Pies":    {"aisle": "Aisle 2", "emoji": "🧴"},
    "Maiz en lata":{"aisle": "Aisle 3", "emoji": "🌽"},
    "Control":     {"aisle": "Aisle 1", "emoji": "🎮"},
    "mouse":       {"aisle": "Aisle 2", "emoji": "🖱️"},
    "Mando_play":  {"aisle": "Aisle 1", "emoji": "🕹️"},
}

def main():
    print("[*] Loading your custom trained model (best.pt)...")
    model = YOLO('runs/detect/yolo_control/weights/best.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n==================================")
    print(" 🎥 UCC MARKET - LIVE SCANNER")
    print("==================================")
    print("-> Point the camera at an inventory product.")
    print("-> Press 'Q' to exit.")

    last_detected = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection - only our trained inventory classes will appear
        results = model.predict(frame, conf=0.70, verbose=False)
        annotated_frame = results[0].plot()

        # Get detected class names
        detected_classes = []
        for r in results:
            for c in r.boxes.cls:
                detected_classes.append(model.names[int(c)])

        h, w = annotated_frame.shape[:2]

        if detected_classes:
            product = detected_classes[0]
            info = INVENTORY.get(product, {"aisle": "Unknown", "emoji": "📦"})

            # Green overlay banner at the bottom
            cv2.rectangle(annotated_frame, (0, h - 80), (w, h), (0, 180, 0), -1)
            cv2.putText(annotated_frame,
                        f"DETECTED: {product}  |  {info['aisle']}",
                        (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame,
                        "Product found in inventory!",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 255, 220), 1)
        else:
            # Red overlay banner - nothing from inventory detected
            cv2.rectangle(annotated_frame, (0, h - 80), (w, h), (0, 0, 180), -1)
            cv2.putText(annotated_frame,
                        "No inventory item detected",
                        (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame,
                        "Point camera at: Atun, Deo Pies or Maiz en lata",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        cv2.imshow("UCC Market AI - Inventory Scanner", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

