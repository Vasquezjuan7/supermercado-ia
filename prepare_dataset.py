import os
import shutil
import random
import yaml

def prepare():
    raw_dir = "dataset_raw"
    train_img_dir = "dataset/train/images"
    val_img_dir = "dataset/val/images"
    train_lbl_dir = "dataset/train/labels"
    val_lbl_dir = "dataset/val/labels"

    # Create dataset directories if they don't exist
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(raw_dir):
        print("No dataset_raw folder found.")
        return

    # Find all product folders inside dataset_raw
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    # Remove 'pip_install_opencv-python' if it exists by accident
    if 'pip_install_opencv-python' in classes:
        classes.remove('pip_install_opencv-python')
        
    if not classes:
        print("No products found in dataset_raw.")
        return

    print(f"[*] Found products to train: {classes}")

    # 1. Update data.yaml automatically with the new classes!
    yaml_data = {
        'path': os.path.abspath('dataset').replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print("[*] Automatically updated 'data.yaml' with new classes!")

    # 2. Clear old dataset to avoid mixing old data
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        for file in os.listdir(d):
            filepath = os.path.join(d, file)
            if os.path.isfile(filepath):
                os.remove(filepath)

    # 3. Copy images and auto-label them using your centered box trick!
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(raw_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        random.shuffle(images)
        
        # Split: 80% of photos for training, 20% for validation (testing)
        split_idx = max(1, int(len(images) * 0.8)) if len(images) > 1 else 1
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:] if len(images) > 1 else []

        # Your clever auto-label format: class_id x_center y_center width height
        label_content = f"{class_id} 0.5 0.5 0.7 0.7\n"

        # Copy train images
        for img_list, img_dest, lbl_dest in [(train_imgs, train_img_dir, train_lbl_dir), 
                                             (val_imgs, val_img_dir, val_lbl_dir)]:
            for img in img_list:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(img_dest, img)
                shutil.copy2(src_path, dst_path)
                
                # Write label .txt file
                label_name = img.replace('.jpg', '.txt')
                with open(os.path.join(lbl_dest, label_name), 'w') as f:
                    f.write(label_content)

        print(f"[+] Processed '{class_name}': {len(train_imgs)} train images, {len(val_imgs)} val images.")

    print("\n[+] Dataset preparation complete! You can now run: python train_control.py")

if __name__ == "__main__":
    prepare()
