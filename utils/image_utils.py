import numpy as np
import matplotlib.pyplot as plt

def resize_image(img, new_size=(48, 48)):
    if img.ndim != 2:
        raise ValueError(f"resize_image expects 2D array, got shape {img.shape}")

    old_h, old_w = img.shape
    new_h, new_w = new_size
    resized = np.empty((new_h, new_w), dtype=img.dtype)

    # ratios
    row_ratio = old_h / new_h
    col_ratio = old_w / new_w

    for i in range(new_h):
        src_i = int(i * row_ratio)
        if src_i >= old_h:
            src_i = old_h - 1
        for j in range(new_w):
            src_j = int(j * col_ratio)
            if src_j >= old_w:
                src_j = old_w - 1
            resized[i, j] = img[src_i, src_j]

    return resized

# Convert RGB image to grayscale using perception weighted formula
def rgb_to_grayscale(img_rgb):
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"rgb_to_grayscale expects (H,W,3), got {img_rgb.shape}")
    # R,G,B indices 0,1,2 respectively
    return 0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]

def bgr_to_grayscale(img_bgr):
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"bgr_to_grayscale expects (H,W,3), got {img_bgr.shape}")
    B = img_bgr[:, :, 0]
    G = img_bgr[:, :, 1]
    R = img_bgr[:, :, 2]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

def _to_float01(arr):
    a = arr.astype(np.float32, copy=False)
    if np.nanmax(a) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)

def load_and_process_image_array(img_array, size=(48, 48)):
    if img_array.ndim == 3:
        if img_array.shape[2] == 3:
            gray = rgb_to_grayscale(img_array)
        elif img_array.shape[2] == 4:  
            gray = rgb_to_grayscale(img_array[:, :, :3])
        else:
            raise ValueError(f"Unsupported 3D image shape: {img_array.shape}")
    elif img_array.ndim == 2:
        gray = img_array
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    gray = _to_float01(gray)
    gray = resize_image(gray, size)
    # Ensure float32 output
    return gray.astype(np.float32)

def load_and_process_image(path, size=(48, 48)):
    img = plt.imread(path)
    if img.ndim == 3:
        if img.shape[2] == 3:
            gray = rgb_to_grayscale(img)           
        elif img.shape[2] == 4:
            gray = rgb_to_grayscale(img[:, :, :3])
        else:
            raise ValueError(f"Unsupported image channels in file: {img.shape}")
    elif img.ndim == 2:
        gray = img                                
    else:
        raise ValueError(f"Unsupported image shape from file: {img.shape}")

    gray = _to_float01(gray)
    gray = resize_image(gray, size)
    return gray.astype(np.float32)

def to_model_input_from_bgr(frame_bgr, size=(48, 48)):
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        gray = bgr_to_grayscale(frame_bgr)
    elif frame_bgr.ndim == 2:
        gray = frame_bgr  
    else:
        raise ValueError(f"Unsupported frame shape: {frame_bgr.shape}")

    gray = _to_float01(gray)
    gray = resize_image(gray, size)
    return gray.astype(np.float32)