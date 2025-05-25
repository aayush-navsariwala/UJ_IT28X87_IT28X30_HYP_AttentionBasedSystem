import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, label):
    images = []
    labels =[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize((48, 48))
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return images, labels
    
def prepare_dataset(split):
    base_path = f'data/processed/{split}/'
    X = []
    y =[]
    
    attentive_images, attentive_labels = load_images_from_folder(os.path.join(base_path, 'attentive'), 1)
    inattentive_images, inattentive_labels = load_images_from_folder(os.path.join(base_path, 'inattentive'), 0)
    
    X.extend(attentive_images + inattentive_images)
    y.extend(attentive_labels + inattentive_labels)
    
    X = np.array(X).reshape(-1, 48, 48)
    y = np.array(y)
    return X, y

def main():
    os.makedirs('data/npy', exist_ok=True)
    
    X_train, y_train = prepare_dataset('train')
    X_test, y_test = prepare_dataset('test')
    
    np.save('data/npy/X_train.npy', X_train)
    np.save('data/npy/y_train.npy', y_train)
    np.save('data/npy/X_test.npy', X_test)
    np.save('data/npy/y_test.npy', y_test)
    
    print(f"Saved: {len(X_train)} training samples, {len(X_test)} test samples")
    
if __name__ == "__main__":
    main()