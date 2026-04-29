import cv2
from ultralytics import YOLO
import os

def verify():
    # Load the custom model you just trained
    model_path = 'runs/detect/yolo_control/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"[!] Error: Trained model not found at {model_path}")
        return

    print(f"[*] Loading your custom trained model: {model_path}")
    model = YOLO(model_path)
    
    # Test with a validation image (one the AI did not use for training)
    image_path = "dataset/val/images/Control_26.jpg"
    
    if not os.path.exists(image_path):
        # If that specific image doesn't exist, find any image in val
        val_imgs = os.listdir("dataset/val/images")
        if val_imgs:
            image_path = os.path.join("dataset/val/images", val_imgs[0])
        else:
            print("[!] Error: No images found in dataset/val/images")
            return

    print(f"[*] Analyzing test image: {image_path}")
    img = cv2.imread(image_path)
    results = model.predict(img, imgsz=640)
    
    print("\n==================================")
    print(" 🚀 RESULTS WITH YOUR CUSTOM AI ")
    print("==================================")
    
    detected = []
    for r in results:
        for c in r.boxes.cls:
            detected.append(model.names[int(c)])
            
    if not detected:
        print("-> The AI still could not recognize the object with enough confidence.")
    else:
        print(f"-> SUCCESS! The AI detected: {detected}")
        
    print("\n==================================")

if __name__ == "__main__":
    verify()
