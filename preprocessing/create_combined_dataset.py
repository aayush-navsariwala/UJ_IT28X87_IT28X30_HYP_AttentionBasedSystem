import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import load_and_process_image, resize_image

# Specify the paths
FER_TRAIN_PATH = 'data/processed/train'
DRIVEGAZE_PATH = 'data/drivegaze'
COMBINED_PATH = 'data/combined'

# Specifying the image size
IMAGE_SIZE      = (48, 48)
TRAIN_RATIO     = 0.8                        
CLEAN_DEST      = True                       
BALANCE_CLASSES = True                       
MAX_PER_CLASS   = None                       
SEED            = 42

VALID_EXTS = ('.jpg', '.jpeg', '.png')

def ensure_clean_dir(path, clean=True):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
def collect_images_flat(folder):
    # Return absolute file paths for images directly inside folder
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTS)]

def load_class_paths():
    # FER-2013 dataset
    fer_att = collect_images_flat(os.path.join(FER_TRAIN_PATH, 'attentive'))
    fer_inatt = collect_images_flat(os.path.join(FER_TRAIN_PATH, 'inattentive'))
    
    # DriveGaze dataset
    dg_att = collect_images_flat(os.path.join(DRIVEGAZE_PATH, 'attentive'))
    dg_inatt = collect_images_flat(os.path.join(DRIVEGAZE_PATH, 'inattentive'))
    
    # Merge sources
    att = fer_att + dg_att
    inatt = fer_inatt + dg_inatt
    
    return att, inatt

def maybe_cap(paths, cap):
    if cap is None or cap <= 0:
        return paths
    return paths[:cap]

def stratified_split(paths, train_ratio):
    # Deterministic split for a single class
    n = len(paths)
    split = int(n * train_ratio)
    return paths[:split], paths[split:]

def save_image(img_path, dest_folder, prefix):
    try:
        img = load_and_process_image(img_path, size=None)  
        img = try_face_crop_gray(img)                     
        img = resize_image(img, IMAGE_SIZE)                
        img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        filename = f"{prefix}_" + os.path.basename(img_path)
        full_path = os.path.join(dest_folder, filename)
        plt.imsave(full_path, img_u8, cmap='gray')
    except Exception as e:
        print(f"Failed to save {img_path}: {e}")
        
def try_face_crop_gray(gray, min_size=60):
    g = (gray * 255.0).astype(np.uint8) if gray.max() <= 1.0 + 1e-6 else gray.astype(np.uint8)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(g, scaleFactor=1.2, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return gray  
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.15 * max(w, h))
    H, W = g.shape[:2]
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    return gray[y1:y2, x1:x2]

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Load all candidate files filtered by DriveGaze extractor
    att_all, inatt_all = load_class_paths()
    print(f"Found attentive: {len(att_all)}")
    print(f"Found inattentive: {len(inatt_all)}")
    
    # Shuffle 
    random.shuffle(att_all)
    random.shuffle(inatt_all)
    
    # Optional cap per class before balancing
    att_all = maybe_cap(att_all, MAX_PER_CLASS)
    inatt_all = maybe_cap(inatt_all, MAX_PER_CLASS)
    
    # Balance classes by downsampling majority
    if BALANCE_CLASSES:
        n = min(len(att_all), len(inatt_all))
        att_all = att_all[:n]
        inatt_all = inatt_all[:n]
        print(f"Balanced to {n} per class.")
        
    # Stratified split per class
    att_train, att_test = stratified_split(att_all, TRAIN_RATIO)
    inatt_train, inatt_test = stratified_split(inatt_all, TRAIN_RATIO)
    
    print(f"Train counts → attentive: {len(att_train)}, inattentive: {len(inatt_train)}")
    print(f"Test  counts → attentive: {len(att_test)},  inattentive: {len(inatt_test)}")
    
    # Prepare destination folders
    if CLEAN_DEST:
        ensure_clean_dir(COMBINED_PATH, clean=True)
    for sub in ['train/attentive','train/inattentive','test/attentive','test/inattentive']:
        os.makedirs(os.path.join(COMBINED_PATH, sub), exist_ok=True)
        
    # Save the images
    for i, p in enumerate(att_train):
        save_image(p, os.path.join(COMBINED_PATH, 'train/attentive'),   f'att_{i}')
    for i, p in enumerate(inatt_train):
        save_image(p, os.path.join(COMBINED_PATH, 'train/inattentive'), f'inatt_{i}')
    for i, p in enumerate(att_test):
        save_image(p, os.path.join(COMBINED_PATH, 'test/attentive'),    f'att_{i}')
    for i, p in enumerate(inatt_test):
        save_image(p, os.path.join(COMBINED_PATH, 'test/inattentive'),  f'inatt_{i}')
        
    print("✅ Combined dataset created at 'data/combined/'")    
    
if __name__ == "__main__":
    main()