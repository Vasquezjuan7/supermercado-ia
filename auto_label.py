import os

def create_placeholder_labels():
    train_labels_dir = "dataset/train/labels"
    val_labels_dir = "dataset/val/labels"
    train_images_dir = "dataset/train/images"
    val_images_dir = "dataset/val/images"
    
    # Label format: class x_center y_center width height
    # 0 (Control) 0.5 0.5 0.7 0.7 (centered at 70% of the image)
    label_content = "0 0.5 0.5 0.7 0.7"
    
    for folder, label_dir in [(train_images_dir, train_labels_dir), (val_images_dir, val_labels_dir)]:
        images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        for img in images:
            label_name = img.replace(".jpg", ".txt")
            with open(os.path.join(label_dir, label_name), "w") as f:
                f.write(label_content)
    
    print("[+] Labels generated (centered placeholder).")

if __name__ == "__main__":
    create_placeholder_labels()
