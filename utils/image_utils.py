import numpy as np
import matplotlib.pyplot as plt

# Resize image to 48x48 using nearest neighbour interpolation
def resize_image(img, new_size=(48, 48)):
    old_h, old_w = img.shape
    new_h, new_w = new_size
    resized = np.zeros((new_h, new_w))

    row_ratio = old_h / new_h
    col_ratio = old_w / new_w

    for i in range(new_h):
        for j in range(new_w):
            src_i = int(i * row_ratio)
            src_j = int(j * col_ratio)
            resized[i, j] = img[src_i, src_j]

    return resized

# Convert RGB image to grayscale using perception weighted formula
def rgb_to_grayscale(img):
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]

# Load image, convert to grayscale, resize and normalise
def load_and_process_image(path, size=(48, 48)):
    img = plt.imread(path)

    # Convert to grayscale if RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb_to_grayscale(img)
    elif img.ndim != 2:
        raise ValueError("Unsupported image shape")

    # Normalise to 0, 1
    img = resize_image(img, size)
    
    # Normalise image
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        # Clipping if JPEG read
        img = np.clip(img, 0.0, 1.0)
        
    return img