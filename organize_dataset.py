import os
import shutil
import random

def organize_dataset():
    raw_dir = "dataset_raw/Control"
    train_img_dir = "dataset/train/images"
    val_img_dir = "dataset/val/images"
    
    # Get all images from the raw folder
    images = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    # Split 80% train, 20% val (approximately 25 and 5)
    split_index = 25
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    print(f"[*] Copying {len(train_images)} images to train...")
    for img in train_images:
        shutil.copy(os.path.join(raw_dir, img), os.path.join(train_img_dir, img))
        
    print(f"[*] Copying {len(val_images)} images to val...")
    for img in val_images:
        shutil.copy(os.path.join(raw_dir, img), os.path.join(val_img_dir, img))
        
    print("[+] Dataset organization complete.")

if __name__ == "__main__":
    organize_dataset()
