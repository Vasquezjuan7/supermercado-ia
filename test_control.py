import cv2
from ultralytics import YOLO
import os

def test_control():
    print("[*] Loading pre-trained generic model (yolo11n.pt)...")
    model = YOLO('yolo11n.pt')
    
    # Test with the first photo of the Control
    image_path = "dataset_raw/Control/Control_0.jpg"
    
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return

    print(f"[*] Analyzing image: {image_path}")
    
    img = cv2.imread(image_path)
    results = model.predict(img, imgsz=640)
    
    print("\n==================================")
    print(" 🧠 RECOGNITION RESULTS ")
    print("==================================")
    
    detected_items = []
    for r in results:
        for c in r.boxes.cls:
            detected_items.append(model.names[int(c)])
            
    if not detected_items:
        print("-> The generic AI DID NOT RECOGNIZE THE OBJECT.")
    else:
        print(f"-> The AI detected: {detected_items}")
        
    print("\n==================================")
    print("💡 CONCLUSION:")
    print("If the AI detected 'remote' (TV remote), that is a good sign.")
    print("If it detected nothing or something incorrect, that is normal.")
    print("The generic AI does not know this specific product. Train it!")
    print("==================================")

if __name__ == "__main__":
    test_control()
