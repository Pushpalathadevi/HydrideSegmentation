import os
import shutil
from PIL import Image

# === User Configurable Paths ===
input_folder = r"V:\maniBackUp\hydride_data"
output_folder = r"V:\maniBackUp\allHydridedImages"
image_list_file = os.path.join(input_folder, "imageNames.txt")

# === Constants ===
MAX_WIDTH = 1024
MAX_HEIGHT = 1024

# === Utility Functions ===

def flatten_filename(relative_path: str) -> str:
    """Replace folder separators with underscores and ensure .png extension."""
    base = relative_path.replace("\\", "_").replace("/", "_")
    return os.path.splitext(base)[0] + ".png"

def process_image(image_path: str) -> Image.Image:
    """
    Load image, convert to grayscale, resize to max 1024x1024 while keeping aspect ratio.
    Returns a PIL Image object.
    """
    img = Image.open(image_path)

    # Convert to grayscale (mode 'L')
    if img.mode != 'L':
        img = img.convert('L')

    # Resize only if necessary
    if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

    return img

def process_all_images():
    """Copy and normalise images listed in *imageNames.txt* to one folder."""
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read image list and process
    with open(image_list_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            relative_path = line.strip()
            if not relative_path:
                continue

            # Construct paths
            source_path = os.path.join(input_folder, relative_path)
            flattened_name = flatten_filename(relative_path)
            dest_path = os.path.join(output_folder, flattened_name)

            # Check file existence and process
            if os.path.exists(source_path):
                try:
                    img = process_image(source_path)
                    img.save(dest_path, format='PNG')
                    print(f"✅ Saved: {dest_path}")
                except Exception as e:
                    print(f"❌ Failed processing {source_path}: {e}")
            else:
                print(f"⚠️ File missing: {source_path}")

# === Run ===
if __name__ == "__main__":
    process_all_images()
