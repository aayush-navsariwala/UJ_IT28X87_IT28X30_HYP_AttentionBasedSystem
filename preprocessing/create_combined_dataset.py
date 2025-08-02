import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.image_utils import load_and_process_image


# Specify the paths
FER_TRAIN_PATH = 'data/processed/train'
DRIVEGAZE_PATH = 'data/drivegaze'
COMBINED_PATH = 'data/combined'

# Specifying the image size
IMAGE_SIZE = (48, 48)

# Split the ratio between 80% to train and 20% to test the model
SPLIT_RATIO = 0.8 

# Function to collect image paths from a label folder
def collect_images(source_folder, label_name):
    full_paths = []
    label_folder = os.path.join(source_folder, label_name)
    
    # Loop through all image files in the folder
    for file in os.listdir(label_folder):
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            full_paths.append((os.path.join(label_folder, file), label_name))
    return full_paths

# Function to load, process and save an image
def save_image(image_path, dest_folder, prefix):
    try:
        # Load the image, convert to grayscale, resize to 48x48 and normalise to [0, 1]
        img = load_and_process_image(image_path)
        
        # Scale pixel values back to [0, 255]
        img = (img * 255).astype(np.uint8)
        
        # Generate a new filename with a prefix to prevent duplicates
        filename = prefix + "_" + os.path.basename(image_path)
        
        # Create the full path to save the image
        full_path = os.path.join(dest_folder, filename)
        
        # Save the image using matplotlib with a grayscale colour map
        plt.imsave(full_path, img, cmap='gray')
    except Exception as e:
        # Error handling if processing or saving fails
        print(f"Failed to save {image_path}: {e}")

def main():
    # Get all image paths from FER-2013 and DriveGaze
    attentive = collect_images(FER_TRAIN_PATH, 'attentive') + collect_images(DRIVEGAZE_PATH, 'attentive')
    inattentive = collect_images(FER_TRAIN_PATH, 'inattentive') + collect_images(DRIVEGAZE_PATH, 'inattentive')
    
    # Shuffle the order of the dataset to mix up data
    random.shuffle(attentive)
    random.shuffle(inattentive)
    
    # Split into train and test
    def split_data(data_list):
        split_point = int(len(data_list) * SPLIT_RATIO)
        return data_list[:split_point], data_list[split_point:]
    
    att_train, att_test = split_data(attentive)
    inatt_train, inatt_test = split_data(inattentive)
    
    # Define the output folders
    folders = ['train/attentive', 'train/inattentive',
               'test/attentive', 'test/inattentive']
    
    for f in folders:
        os.makedirs(os.path.join(COMBINED_PATH, f), exist_ok=True)
        
    # Save all of the images
    for i, (img_path, _) in enumerate(att_train):
        save_image(img_path, os.path.join(COMBINED_PATH, 'train/attentive'), f'att_{i}')
    for i, (img_path, _) in enumerate(att_test):
        save_image(img_path, os.path.join(COMBINED_PATH, 'test/attentive'), f'att_{i}')
    for i, (img_path, _) in enumerate(inatt_train):
        save_image(img_path, os.path.join(COMBINED_PATH, 'train/inattentive'), f'inatt_{i}')
    for i, (img_path, _) in enumerate(inatt_test):
        save_image(img_path, os.path.join(COMBINED_PATH, 'test/inattentive'), f'inatt_{i}')
        
    print("Combined dataset created at 'data/combined/'")
    
if __name__ == "__main__":
    main()