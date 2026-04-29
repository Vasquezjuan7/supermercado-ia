import cv2
import os

def main():
    save_dir = "dataset_raw"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n=============================================")
    print(" 🛒 UCC Market AI - Dataset Capture Tool")
    print("=============================================\n")
    product_name = input("What product are you going to scan? (e.g. milk, cookies): ").strip().replace(" ", "_")
    if not product_name:
        print("Product name cannot be empty.")
        return

    product_dir = os.path.join(save_dir, product_name)
    if not os.path.exists(product_dir):
        os.makedirs(product_dir)

    print(f"\n[INSTRUCTIONS]")
    print(f"1. Your camera will open.")
    print(f"2. Place the product '{product_name}' in front of the lens.")
    print(f"3. Press SPACEBAR to take a photo (Take about 50 photos rotating the object).")
    print(f"4. Press the 'Q' key to exit when you are done.\n")
    
    print(f"\n[CAMERA SETTINGS]")
    print("0: Laptop Webcam (default)")
    print("1: Secondary Camera")
    print("URL: IP Camera (e.g., http://192.168.1.15:8080/video)")
    camera_source = input("\nEnter camera ID or URL [0]: ").strip()
    
    # Si es un número (ID), lo convertimos a int. Si no, se queda como string (URL).
    if not camera_source:
        camera_source = 0
    elif camera_source.isdigit():
        camera_source = int(camera_source)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source: {camera_source}")
        return

    count = 0
    existing_files = os.listdir(product_dir)
    if existing_files:
        count = len(existing_files)

    print("Loading camera... Click the camera window if it doesn't respond.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            break

        frame_clean = frame.copy()

        info_text = f"PRODUCT: {product_name} | PHOTOS: {count}"
        cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Take Photo | Q: Quit", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("UCC Market - Training AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32: # SPACE
            img_path = os.path.join(product_dir, f"{product_name}_{count}.jpg")
            cv2.imwrite(img_path, frame_clean)
            print(f"[*] Saved: {img_path}")
            count += 1
        elif key == 27 or key == ord('q'): # ESC or Q
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Taken {count} photos for '{product_name}'. You can find them in 'ia-service/{product_dir}'.")
    print("To scan another object, run the script again.")

if __name__ == "__main__":
    main()
