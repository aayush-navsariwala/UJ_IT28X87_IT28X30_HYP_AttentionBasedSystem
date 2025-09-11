import os
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import load_and_process_image

SEED = 42
VALID_EXTS = ('.jpg', '.jpeg', '.png')

def load_images_from_folder(folder_path, label):
    images, labels = [], []
    if not os.path.isdir(folder_path):
        print(f"[warn] missing folder: {folder_path}")
        return images, labels

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(VALID_EXTS):
            continue
        img_path = os.path.join(folder_path, filename)
        try:
            img = load_and_process_image(img_path)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"[err] loading {img_path}: {e}")
    return images, labels

def prepare_dataset(split):
    base_path = os.path.join('data', 'combined', split)
    att_imgs, att_labels = load_images_from_folder(os.path.join(base_path, 'attentive'), 1)
    ina_imgs, ina_labels = load_images_from_folder(os.path.join(base_path, 'inattentive'), 0)

    X_list = att_imgs + ina_imgs
    y_list = att_labels + ina_labels

    X = np.asarray(X_list, dtype=np.float32).reshape(-1, 48, 48)
    y = np.asarray(y_list, dtype=np.int32)

    # Sanity checks
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"[fatal] Found NaN/Inf in X for split={split}")
    if X.size > 0 and (X.min() < -1e-6 or X.max() > 1.0 + 1e-6):
        print(f"[warn] X not in [0,1]? min={X.min():.3f} max={X.max():.3f} (split={split})")

    rng = np.random.default_rng(SEED)
    if len(y) > 0:
        idx = rng.permutation(len(y))
        X = np.ascontiguousarray(X[idx])  
        y = y[idx]

    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    print(f"[{split}] total={len(y)} | att={pos} | inatt={neg}")

    return X, y

def main():
    os.makedirs('data/npy', exist_ok=True)

    # Required splits
    X_train, y_train = prepare_dataset('train')
    X_test,  y_test  = prepare_dataset('test')

    # Optional validation split (if you created data/combined/val)
    has_val = os.path.isdir(os.path.join('data', 'combined', 'val'))
    if has_val:
        X_val, y_val = prepare_dataset('val')

    # Save
    np.save('data/npy/X_train.npy', X_train)
    np.save('data/npy/y_train.npy', y_train)
    np.save('data/npy/X_test.npy',  X_test)
    np.save('data/npy/y_test.npy',  y_test)
    if has_val:
        np.save('data/npy/X_val.npy', X_val)
        np.save('data/npy/y_val.npy', y_val)

    # Summary
    print(
        f"Saved: train={len(X_train)} | test={len(X_test)}"
        + (f" | val={len(X_val)}" if has_val else "")
    )
    print(
        f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}"
        + (f", X_val={X_val.shape}, y_val={y_val.shape}" if has_val else "")
    )
    
# Entry point to execute the script
if __name__ == "__main__":
    main()