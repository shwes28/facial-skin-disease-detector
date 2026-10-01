"""
Data Cleaning & 1:1 Balancing Pipeline
--------------------------------------
1. Validates & cleans raw images (removes corrupt/non-RGB/tiny files).
2. Performs balanced under-sampling to guarantee zero class imbalance (50% / 50%).
3. Distributes images into data/train and data/val (80/20 split).
"""

import os
import shutil
import random
from PIL import Image

RAW_ACNE_DIR = "raw_data/acne"
RAW_OTHER_DIR = "raw_data/other"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.8  # 80% train, 20% validation
MIN_RESOLUTION = (100, 100)

def is_valid_image(filepath, min_size=MIN_RESOLUTION):
    """Verifies file integrity, minimum resolution, and RGB capability."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            if img.width < min_size[0] or img.height < min_size[1]:
                return False
            img.convert("RGB")
        return True
    except Exception:
        return False

def clean_and_collect_images(source_dir):
    """Scans directory and returns list of verified valid image paths."""
    if not os.path.exists(source_dir):
        return []
    valid_images = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(supported_extensions):
                full_path = os.path.join(root, f)
                if is_valid_image(full_path):
                    valid_images.append(full_path)
                else:
                    print(f"[SKIPPED] Invalid or corrupt image: {f}")
    return valid_images

def balance_and_split():
    print("=" * 60)
    print("1. SCANNING & VALIDATING RAW IMAGES")
    print("=" * 60)
    clean_acne = clean_and_collect_images(RAW_ACNE_DIR)
    clean_other = clean_and_collect_images(RAW_OTHER_DIR)
    
    print(f"Verified clean Acne images: {len(clean_acne)}")
    print(f"Verified clean Other Lesion images: {len(clean_other)}")
    
    if len(clean_acne) == 0 or len(clean_other) == 0:
        print("\n[NOTE] Raw directories are currently empty or not found.")
        print(f"Place raw images inside '{RAW_ACNE_DIR}' and '{RAW_OTHER_DIR}' to balance.")
        return

    # Enforce exact 1:1 balance
    target_count = min(len(clean_acne), len(clean_other))
    print(f"\n[BALANCING] Selecting exactly {target_count} images per class (50% / 50% split).")
    
    random.seed(42)  # For reproducible sampling
    balanced_acne = random.sample(clean_acne, target_count)
    balanced_other = random.sample(clean_other, target_count)
    
    train_count = int(target_count * TRAIN_RATIO)
    val_count = target_count - train_count
    
    splits = {
        'train': {'acne': balanced_acne[:train_count], 'other': balanced_other[:train_count]},
        'val': {'acne': balanced_acne[train_count:], 'other': balanced_other[train_count:]}
    }
    
    for split_name, categories in splits.items():
        for category, file_list in categories.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, category)
            os.makedirs(dest_dir, exist_ok=True)
            for src_path in file_list:
                shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))
            print(f"Saved {len(file_list)} images to {dest_dir}")
            
    print("\n[SUCCESS] Balanced dataset created!")
    print(f"  Train: {train_count * 2} images (50% acne / 50% other)")
    print(f"  Val:   {val_count * 2} images (50% acne / 50% other)")

if __name__ == "__main__":
    balance_and_split()
