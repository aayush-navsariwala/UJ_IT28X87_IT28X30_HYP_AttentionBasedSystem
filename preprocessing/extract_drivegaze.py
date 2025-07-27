import os
import shutil
from PIL import Image

# Path to the DriveGaze dataset folder on local machine
SOURCE_DIR = r'C:\Users\aayus\Desktop\UJ\Year Project\HYP\drivegaze\DriveGaze\frame'

# Destination folder in project file structure
DEST_DIR = 'data/drivegaze'

# Resizing images to match FER-2013 dataset
IMAGE_SIZE = (48, 48)

# Attention labelling based on DriveGaze dataset
ATTENTIVE = {'focus', 'excited'}
INATTENTIVE = {'angry', 'brake', 'distracted', 'mistake', 'tired'}

#  Processes a single label folder from the DriveGaze dataset
def process_label_folder(label_name, label_type):
    label_path = os.path.join(SOURCE_DIR, label_name)
    output_folder = os.path.join(DEST_DIR, label_type)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing label: {label_name} → {label_type}")
    
    # Go through each sequence inside the label folder
    for seq_folder in os.listdir(label_path):
        seq_path = os.path.join(label_path, seq_folder)
        
        # Skips anything that is not a folder
        if not os.path.isdir(seq_path):
            continue
        
        # Process each image inside the sequence folder
        for image_name in os.listdir(seq_path):
            if not image_name.lower().endswith('.jpg'):
                continue
            
            image_path = os.path.join(seq_path, image_name)
            
            try:
                # Loads the image, converts it to grayscale, resizes it to 48x48
                img = Image.open(image_path).convert('L')
                img = img.resize(IMAGE_SIZE)
                
                # Save the processed image with a unique name in the destination folder
                output_path = os.path.join(output_folder, f"{label_name}_{seq_folder}_{image_name}")
                img.save(output_path)
            
            except Exception as e:
                # Error logging
                print(f"Failed to process {image_path}: {e}")
                
def main():
    for label in ATTENTIVE:
        process_label_folder(label, 'attentive')
    for label in INATTENTIVE:
        process_label_folder(label, 'inattentive')
    print("DriveGaze extraction complete")
    
if __name__ == "__main__":
    main()