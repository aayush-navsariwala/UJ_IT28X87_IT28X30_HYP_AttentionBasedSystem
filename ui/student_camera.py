import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import numpy as np
from utils.image_utils import resize_image, rgb_to_grayscale
from model.cnn import SimpleCNN

# Initialise the model
model = SimpleCNN()

# Indexing the camera
camera = cv2.VideoCapture(0)

# Check if the camera opened
if not camera.isOpened():
    print("Cannot access webcam.")
    exit()
    
print("Student camera active. Capturing every 5 seconds...")

try:
    while True:
        # Read a frame from the webcam
        ret, frame = camera.read()
        
        if not ret:
            print("Failed to capture image")
            continue
        
        # Resize the frame for consistency
        frame = cv2.resize(frame, (200, 200))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Manually grayscale
        gray = rgb_to_grayscale(frame_rgb)
        
        # Resize to 48x48
        gray_resized = resize_image(gray, (48, 48))
        
        # Normalise
        img = gray_resized / 255.0
        
        # Predict attention
        prediction = model.predict(img)
        
        # Display result
        status = "Attentive" if prediction == 1 else "Inattentive"
        print(f"[{time.strftime('%H:%M:%S')}] {status}")
        
        # Wait 5 seconds before the next capture
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n Camera monitoring stopped.")
    
finally:
    camera.release()
    cv2.destroyAllWindows()