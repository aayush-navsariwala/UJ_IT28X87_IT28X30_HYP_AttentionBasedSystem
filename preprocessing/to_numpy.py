import os
import numpy as np
from PIL import Image

# Loads all images from a specified folder and do various things with them
def load_images_from_folder(folder_path, label):
    images = []
    labels =[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            
            # Open the image and convert it to grayscale
            img = Image.open(img_path).convert('L')
            
            # Resize the image to 48x48 to match the FER2013 dataset
            img = img.resize((48, 48))
            
            # Convert to NumPy array and normalise pixel values 
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return images, labels

# Prepares either the train or test dataset by loading images from attentive and inattentive folders
def prepare_dataset(split):
    base_path = f'data/processed/{split}/'
    # Image data
    X = []
    
    # Labels
    y =[]
    
    # Load images and labels for attentive (1) and inattentive (2)
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
    print(f"Saved: {len(X_train)} training samples, {len(X_test)} test samples")
    
# Entry point to execute the script
if __name__ == "__main__":
    main()