import os
import cv2
import numpy as np
from pathlib import Path

# Set this to the root folder of the Sadhana dataset
DATASET_PATH = r"c:\final year Project\ml\dataset\sadhana"

def prepare_dataset():
    """
    Scans the Sadhana dataset folder structure.
    Reads all character images from each subfolder.
    Uses the folder name as the label for each image.
    Resizes every image to height=64, width=256, grayscale.
    Saves a final dataset as dataset_images.npy and dataset_labels.txt.
    """
    root_dir = Path(DATASET_PATH)
    
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"Error: Directory '{DATASET_PATH}' does not exist.")
        print("Please update DATASET_PATH at the top of the script.")
        return

    images = []
    labels = []
    
    # Subfolders represent the labels (characters)
    subfolders = [f for f in root_dir.iterdir() if f.is_dir()]
    
    print(f"Scanning {len(subfolders)} subfolders in {DATASET_PATH}...")
    
    for folder in subfolders:
        label = folder.name
        
        # Valid image extensions
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_files.extend(list(folder.glob(ext)))
            
        for img_path in image_files:
            try:
                # Read grayscale
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize to (width=256, height=64)
                resized_img = cv2.resize(img, (256, 64))
                
                images.append(resized_img)
                labels.append(label)
                
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                
    if not images:
        print("No images found! Please check the dataset path and format.")
        return
        
    # Convert to numpy array
    # Shape will be (N, 64, 256) where N is number of images
    np_images = np.array(images, dtype=np.uint8)
    
    # Save the files
    print("Saving dataset_images.npy...")
    np.save("dataset_images.npy", np_images)
    
    print("Saving dataset_labels.txt...")
    with open("dataset_labels.txt", "w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")
            
    # Print total count
    print(f"\n✅ Processing Complete!")
    print(f"Total image count: {len(images)}")
    print("Saved 'dataset_images.npy' and 'dataset_labels.txt'")

if __name__ == "__main__":
    prepare_dataset()
