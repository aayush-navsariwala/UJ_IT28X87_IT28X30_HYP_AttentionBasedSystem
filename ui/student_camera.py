# Added parent directory to path for debugging local imports
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

# Set up webcam capture using the default camera
camera = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not camera.isOpened():
    print("Cannot access webcam.")
    exit()
    
print("Student camera active. Capturing every 5 seconds...")

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = camera.read()
        
        # Handle capture failure
        if not ret:
            print("Failed to capture image")
            continue
        
        # Resize the frame for standard preview size
        frame = cv2.resize(frame, (200, 200))
        
        # Convert OpenCV BGR format to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert RGB image to grayscale manually using formula in image_utils.py
        gray = rgb_to_grayscale(frame_rgb)
        
        # Resize image to CNN input size of 48x48
        gray_resized = resize_image(gray, (48, 48))
        
        # Normalise pixel values to range [0, 1]
        img = gray_resized / 255.0
        
        # Predict attention using model
        prediction = model.predict(img)
        
        # Display prediction result with timestamp
        status = "Attentive" if prediction == 1 else "Inattentive"
        print(f"[{time.strftime('%H:%M:%S')}] {status}")
        
        # Wait 5 seconds before the next capture
        time.sleep(5)

# Allow application exit through Ctrl+C
except KeyboardInterrupt:
    print("\n Camera monitoring stopped.")

# Release the camera and any OpenCV windows
finally:
    camera.release()
    cv2.destroyAllWindows()