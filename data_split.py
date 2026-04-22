import os
import shutil
import random
from pathlib import Path

# Since you are running this from C:\Users\ansif\Downloads\Jezt
# Your raw data is inside the "Images" folder.
RAW_DATA_DIR = Path("Images") 
TRAIN_DIR = Path("dataset/train")
EVAL_DIR = Path("dataset/eval")
SPLIT_RATIO = 0.8  # 80% train, 20% eval

def prepare_dataset():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Cannot find the folder: {RAW_DATA_DIR.absolute()}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    identities = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    total_train = 0
    total_eval = 0

    print("Initiating deep-scan data split...")

    for identity_path in identities:
        identity_name = identity_path.name
        
        # THE FIX: .rglob() searches deeply into ALL subfolders (like High_quality)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(identity_path.rglob(ext)))
        
        if len(images) < 2:
            print(f"  [!] Skipping '{identity_name}': Only {len(images)} images found.")
            continue

        # Strict randomization
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        
        train_imgs = images[:split_idx]
        eval_imgs = images[split_idx:]

        # Create identity folders
        (TRAIN_DIR / identity_name).mkdir(exist_ok=True)
        (EVAL_DIR / identity_name).mkdir(exist_ok=True)

        # Rename files safely so they don't overwrite each other when flattened
        for i, img in enumerate(train_imgs):
            safe_name = f"{identity_name}_train_{i}{img.suffix.lower()}"
            shutil.copy(img, TRAIN_DIR / identity_name / safe_name)
            
        for i, img in enumerate(eval_imgs):
            safe_name = f"{identity_name}_eval_{i}{img.suffix.lower()}"
            shutil.copy(img, EVAL_DIR / identity_name / safe_name)

        print(f"  -> Processed '{identity_name}': {len(train_imgs)} Train | {len(eval_imgs)} Eval")
        total_train += len(train_imgs)
        total_eval += len(eval_imgs)

    print("-" * 40)
    print(f"Dataset successfully compiled and flattened.")
    print(f"Total Training Images: {total_train}")
    print(f"Total Evaluation Images: {total_eval}")

if __name__ == "__main__":
    prepare_dataset()