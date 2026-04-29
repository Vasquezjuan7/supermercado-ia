import cv2
from ultralytics import YOLO
import sys

def main():
    print("[*] Loading pre-trained generic model (yolo11n.pt)...")
    model = YOLO('yolo11n.pt')
    
    image_path = "dataset_raw/talco_pies/talco_pies_0.jpg"
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] Error: Could not load image {image_path}")
            return
    except Exception as e:
        print(f"[!] File not found: {e}")
        return

    print(f"[*] Analyzing image: {image_path}")
    
    results = model.predict(img, imgsz=640)
    
    print("\n==================================")
    print(" 🧠 CURRENT AI RECOGNITION RESULTS ")
    print("==================================")
    
    detected_items = []
    for r in results:
        for c in r.boxes.cls:
            detected_items.append(model.names[int(c)])
            
    if not detected_items:
        print("-> The generic AI DID NOT RECOGNIZE ANY OBJECT. (It looks like an empty background to it)")
    else:
        print(f"-> The generic AI thinks it saw: {detected_items}")
        
    print("\n==================================")
    print("💡 CONCLUSION:")
    print("As you can see, the factory pre-trained AI only knows general objects (e.g., bottles, people, chairs).")
    print("If it failed to detect 'talco_pies', this is EXACTLY why we must train it")
    print("using the 30 photos and perform Custom Training!")
    print("==================================")

if __name__ == "__main__":
    main()
