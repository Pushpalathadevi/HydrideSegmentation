import subprocess
import os
from tqdm import tqdm

def export_hydride_mask_with_gimp(xcf_filepath):
    if not os.path.isfile(xcf_filepath):
        print(f"File not found: {xcf_filepath}")
        return

    base_name = os.path.splitext(os.path.basename(xcf_filepath))[0]

    # ✅ Change 1: Fixed export directory
    export_dir = r"L:\maniBackUp\HydrideSegmentation\HydrideMask_Manual\dataset_raw"
    os.makedirs(export_dir, exist_ok=True)

    # ✅ Change 2: Keep same base name, different extensions
    output_png = os.path.join(export_dir, f"{base_name}.png")
    output_jpg = os.path.join(export_dir, f"{base_name}.jpg")

    xcf_path_gimp = xcf_filepath.replace("\\", "/")
    output_png_gimp = output_png.replace("\\", "/")
    output_jpg_gimp = output_jpg.replace("\\", "/")

    # ✅ Change 3: Extend GIMP batch script to also export background layer
    gimp_batch_script = f'''
from gimpfu import *

def find_layer_by_name(layers, name):
    for layer in layers:
        if layer.name == name:
            return layer
        if pdb.gimp_item_is_group(layer):
            found = find_layer_by_name(layer.children, name)
            if found:
                return found
    return None

def export_layers():
    image = pdb.gimp_file_load("{xcf_path_gimp}", "{xcf_path_gimp}")

    # Export hydride_mask layer as PNG
    layer = find_layer_by_name(image.layers, "hydride_mask")
    if layer:
        pdb.gimp_layer_set_opacity(layer, 100.0)
        new_image = pdb.gimp_image_new(layer.width, layer.height, RGB)
        new_layer = pdb.gimp_layer_new_from_drawable(layer, new_image)
        pdb.gimp_image_add_layer(new_image, new_layer, 0)
        pdb.file_png_save(new_image, new_layer, "{output_png_gimp}", "{output_png_gimp}", 0, 9, 0, 0, 0, 0, 0)
        pdb.gimp_image_delete(new_image)
    else:
        pdb.gimp_message("Layer 'hydride_mask' not found")

    # Export background layer as JPG
    layer_bg = find_layer_by_name(image.layers, "background")
    if layer_bg:
        pdb.gimp_layer_set_opacity(layer_bg, 100.0)
        new_image = pdb.gimp_image_new(layer_bg.width, layer_bg.height, RGB)
        new_layer = pdb.gimp_layer_new_from_drawable(layer_bg, new_image)
        pdb.gimp_image_add_layer(new_image, new_layer, 0)
        pdb.file_jpeg_save(new_image, new_layer, "{output_jpg_gimp}", "{output_jpg_gimp}", 0.9, 0, 0, 0, "", 0, 1, 0, 0)
        pdb.gimp_image_delete(new_image)
    else:
        pdb.gimp_message("Layer 'background' not found")

    pdb.gimp_image_delete(image)

export_layers()
pdb.gimp_quit(0)
'''

    command = [
        r"C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe",
        "-i",
        "--batch-interpreter=python-fu-eval",
        "-b", gimp_batch_script,
        "-b", "(gimp-quit 0)"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Exported to folder: {export_dir}")
        print(f"   - Hydride mask (PNG): {output_png}")
        print(f"   - Background (JPG):  {output_jpg}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running GIMP: {e}")

if __name__ == "__main__":
    input_folder = r"L:\maniBackUp\HydrideSegmentation\HydridedImagesForTraining"
    xcf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".xcf")]
    for filename in tqdm(xcf_files, desc="Processing XCF files", unit="file"):
        if filename.lower().endswith(".xcf"):
            xcf_file = os.path.join(input_folder, filename)
            export_hydride_mask_with_gimp(xcf_file)
