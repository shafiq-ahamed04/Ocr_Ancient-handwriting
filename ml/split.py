import os
import shutil
import random

SRC_DIR = "images"
TRAIN_DIR = "train"
VAL_DIR = "val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for i in range(40):  # 0 to 39
    filename = f"{i}.jpg"
    src_path = os.path.join(SRC_DIR, filename)

    if not os.path.exists(src_path):
        print("Missing:", src_path)
        continue

    label = str(i)  # temporary label = image number (we'll fix class labels later)
    train_label_dir = os.path.join(TRAIN_DIR, label)
    val_label_dir = os.path.join(VAL_DIR, label)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # simple split: 80% train, 20% val
    dst_dir = train_label_dir if random.random() < 0.8 else val_label_dir
    shutil.copy(src_path, os.path.join(dst_dir, filename))

print("Split done")
