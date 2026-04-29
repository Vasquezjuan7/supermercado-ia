from ultralytics import YOLO
import os

def train_model():
    print("[*] Starting training for all inventory products with YOLO11 (30 epochs)...")
    
    # Load the YOLO11 base model
    model = YOLO('yolo11n.pt')
    
    # Start training
    results = model.train(
        data='data.yaml',
        epochs=30,        # Increase for better accuracy
        imgsz=640,
        batch=8,
        name='yolo_control',
        exist_ok=True     # Overwrite previous training folder
    )
    
    print("\n[+] Training complete.")
    print("[*] Trained model saved at: runs/detect/yolo_control/weights/best.pt")

if __name__ == "__main__":
    train_model()
