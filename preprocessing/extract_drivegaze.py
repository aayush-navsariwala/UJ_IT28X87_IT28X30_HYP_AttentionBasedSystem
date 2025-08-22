import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils.image_utils import load_and_process_image

# Path to the DriveGaze dataset folder on local machine
SOURCE_DIR = r'C:\Users\aayus\Desktop\UJ\Year Project\HYP\drivegaze\DriveGaze\frame'

# Destination folder in project file structure
DEST_DIR = 'data/drivegaze'

# Resizing images to match FER-2013 dataset
IMAGE_SIZE = (48, 48)

# Attention labelling based on DriveGaze dataset
ATTENTIVE = {'focus'}
INATTENTIVE = {'distracted', 'tired'}

# Clears the previous extracted images to not mix old labels
CLEAN_DEST = True

# If true, just count what would be processed
DRY_RUN = False

def ensure_clean_dir(path):
    if CLEAN_DEST and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
def save_gray_uint8(image_f32, out_path):
    # image_f32: float32 in [0,1], save as 8-bit grayscale png/jpg.
    arr = np.clip(image_f32 * 255.0, 0, 255).astype(np.uint8)
    # Use cmap='gray' to keep it grayscale
    plt.imsave(out_path, arr, cmap='gray')

#  Processes a single label folder from the DriveGaze dataset
def process_label_folder(label_name, label_type):
    label_path = os.path.join(SOURCE_DIR, label_name)
    if not os.path.isdir(label_path):
        print(f"⚠️  Skipping missing label folder: {label_path}")
        return 0
    
    out_dir = os.path.join(DEST_DIR, label_type)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing label: {label_name}  →  {label_type}")
    
    count = 0
    
    for seq_folder in os.listdir(label_path):
        seq_path = os.path.join(label_path, seq_folder)
        if not os.path.isdir(seq_path):
            continue
        
        for image_name in os.listdir(seq_path):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            src = os.path.join(seq_path, image_name)
            try:
                # Pipeline: read -> grayscale -> resize -> normalize [0,1]
                img = load_and_process_image(src, size=IMAGE_SIZE)

                if DRY_RUN:
                    count += 1
                    continue
                
                # Save to destination as grayscale
                out_name = f"{label_name}_{seq_folder}_{image_name}"
                out_path = os.path.join(out_dir, out_name)
                save_gray_uint8(img, out_path)
                count += 1
                
            except Exception as e:
                print(f"Failed to process {src}: {e}")
    return count
                
def main():
    # Prepare destination folders
    if CLEAN_DEST:
        ensure_clean_dir(DEST_DIR)
    os.makedirs(os.path.join(DEST_DIR, 'attentive'), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'inattentive'), exist_ok=True)

    total_att = 0
    total_inatt = 0

    # Only these labels are used, others are ignored entirely
    for label in ATTENTIVE:
        total_att += process_label_folder(label, 'attentive')
    for label in INATTENTIVE:
        total_inatt += process_label_folder(label, 'inattentive')

    print("✅ DriveGaze extraction complete")
    print(f"  Attentive   (focus)      : {total_att}")
    print(f"  Inattentive (distracted,tired): {total_inatt}")
    if DRY_RUN:
        print("  (Dry run: nothing was written)")
    
if __name__ == "__main__":
    main()