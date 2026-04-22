import os
from pathlib import Path
from PIL import Image
import warnings

# Suppress PIL warnings for large/weird images, we just care if they open
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def sanitize_dataset(directory):
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    print(f"Initiating sanitation sweep on: {directory}")
    total_scanned = 0
    corrupted_count = 0

    # Recursively find every file
    for img_path in path.rglob("*.*"):
        if img_path.is_file():
            total_scanned += 1
            try:
                # Try to open and verify the bytes
                with Image.open(img_path) as img:
                    img.verify() # verify() checks the file integrity without loading the whole array
            except Exception as e:
                # If PIL throws ANY error, the file is dead weight. Delete it.
                print(f"  [!] Deleting corrupted file: {img_path}")
                os.remove(img_path)
                corrupted_count += 1

    print(f"Sweep complete for {directory}.")
    print(f"Scanned: {total_scanned} | Vaporized: {corrupted_count}\n")

if __name__ == "__main__":
    print("--- Dataset Sanitation Protocol ---")
    sanitize_dataset("images")
    # sanitize_dataset("dataset/eval")
    print("Dataset is clean. Ready for PyTorch ingestion.")