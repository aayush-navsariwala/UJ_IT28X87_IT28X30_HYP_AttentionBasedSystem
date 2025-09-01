import os
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import load_and_process_image

# Loads all images from a specified folder and do various things with them
def load_images_from_folder(folder_path, label):
    # List to store processed image arrays
    images = []
    # List to store corresponding labels
    labels = []
    for filename in os.listdir(folder_path):
        # Filter for supported file formats
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Build full image path
                img_path = os.path.join(folder_path, filename) 
                
                # Load, grayscale, resize and normalise the image
                img = load_and_process_image(img_path)
                
                # Append processed image and label to respective lists
                images.append(img)
                labels.append(label)
            except Exception as e:
                # Error handling for loading images from dataset
                print(f"Error loading image {img_path}: {e}")
    return images, labels

# Prepares either the train or test dataset by loading images from attentive and inattentive folders
def prepare_dataset(split):
    base_path = f'data/combined/{split}/'
    X, y = [], []
    
    # Load images and labels for attentive (1) and inattentive (0)
    attentive_images, attentive_labels = load_images_from_folder(os.path.join(base_path, 'attentive'), 1)
    inattentive_images, inattentive_labels = load_images_from_folder(os.path.join(base_path, 'inattentive'), 0)
    
    # Combine both attentive and inattentive data
    X.extend(attentive_images + inattentive_images)
    y.extend(attentive_labels + inattentive_labels)
    
    # Convert lists to NumPy arrays
    X = np.array(X).reshape(-1, 48, 48)
    y = np.array(y)
    return X, y

# Main function to preprocess both train and test datasets 
def main():
    # Create output directory if it does not exist
    os.makedirs('data/npy', exist_ok=True)
    
    # Prepare train and test datasets
    X_train, y_train = prepare_dataset('train')
    X_test, y_test = prepare_dataset('test')
    
    # Save datasets as NumPy binary files
    np.save('data/npy/X_train.npy', X_train)
    np.save('data/npy/y_train.npy', y_train)
    np.save('data/npy/X_test.npy', X_test)
    np.save('data/npy/y_test.npy', y_test)
    
    # Output dataset statistics
    print(f"Saved: {len(X_train)} training samples and {len(X_test)} test samples")
    
# Entry point to execute the script
if __name__ == "__main__":
    main()