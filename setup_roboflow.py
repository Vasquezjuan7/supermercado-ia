import zipfile
import shutil
import os

DOWNLOADS = os.path.expanduser("~/Downloads")
IA_SERVICE = os.path.dirname(os.path.abspath(__file__))

# 1. Find the Roboflow zip in Downloads
zip_file = None
for f in os.listdir(DOWNLOADS):
    if f.startswith("Supermarket") and f.endswith(".zip"):
        zip_file = os.path.join(DOWNLOADS, f)
        break

if not zip_file:
    print("[!] Could not find the Roboflow ZIP file in Downloads.")
    print("    Make sure the file starts with 'Supermarket' and ends with '.zip'")
    exit(1)

print(f"[*] Found: {zip_file}")

# 2. Extract to a temp folder
extract_dir = os.path.join(IA_SERVICE, "_roboflow_temp")
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

print("[*] Extracting ZIP...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 3. Show extracted contents
print("\n[*] Contents extracted:")
for item in os.listdir(extract_dir):
    print(f"    -> {item}")

# 4. Replace dataset folder and data.yaml
dataset_dest = os.path.join(IA_SERVICE, "dataset")
if os.path.exists(dataset_dest):
    shutil.rmtree(dataset_dest)
os.makedirs(dataset_dest)

# Copy train folder
for folder_name in ["train", "valid", "test"]:
    src = os.path.join(extract_dir, folder_name)
    if os.path.exists(src):
        dst_name = "val" if folder_name == "valid" else folder_name
        dst = os.path.join(dataset_dest, dst_name)
        shutil.copytree(src, dst)
        print(f"[+] Copied '{folder_name}' -> 'dataset/{dst_name}'")

# 5. Copy and fix data.yaml
yaml_src = os.path.join(extract_dir, "data.yaml")
yaml_dst = os.path.join(IA_SERVICE, "data.yaml")

if os.path.exists(yaml_src):
    with open(yaml_src, "r") as f:
        content = f.read()

    # Fix paths to be absolute and correct
    abs_dataset = dataset_dest.replace("\\", "/")
    content = content.replace("../train", "train/images")
    content = content.replace("../valid", "val/images")
    content = content.replace("../test", "test/images")

    # Write corrected yaml
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("path:"):
            new_lines.append(f"path: {abs_dataset}")
        elif line.startswith("train:"):
            new_lines.append("train: train/images")
        elif line.startswith("val:") or line.startswith("valid:"):
            new_lines.append("val: val/images")
        else:
            new_lines.append(line)

    with open(yaml_dst, "w") as f:
        f.write("\n".join(new_lines))

    print(f"[+] data.yaml updated successfully!")

# 6. Cleanup temp folder
shutil.rmtree(extract_dir)
print("\n[+] Setup complete!")
print("[*] Run: python train_control.py")
print("    -> Your AI will now train with Roboflow's precise annotations!")
