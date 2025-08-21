# Added parent directory to path for debugging local imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from utils.image_utils import rgb_to_grayscale, resize_image
from model.cnn import SimpleCNN

# Helpers for thresholding imports
def load_threshold(default=0.5, path="weights/threshold.txt"):
    t = default
    try:
        with open(path) as f:
            t = float(f.read().strip())
    except FileNotFoundError:
        print(f"[Info] No threshold file at {path}. Using default={default}.")
    except Exception as e:
        print(f"[Warn] Failed to read threshold from {path}: {e}. Using default={default}.")
    return t

def to_model_input_from_bgr(frame_bgr):
    # Convert a BGR frame (OpenCV) to the model's 48x48 grayscale normalized input.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = rgb_to_grayscale(rgb)          
    resized = resize_image(gray, (48, 48))  

    # Normalize once
    if resized.dtype != np.float32:
        resized = resized.astype(np.float32)
    # If values look like 0..255, scale down; if already 0..1, this clip keeps it safe.
    if resized.max() > 1.0:
        resized = resized / 255.0
    else:
        resized = np.clip(resized, 0.0, 1.0)
    return resized

def rolling_mean(deq):
    return float(np.mean(deq)) if len(deq) else 0.0

# Model initialisation
model = SimpleCNN()

# Load trained weights
weights_path = "weights/best.npz"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"[OK] Loaded weights from {weights_path}")
else:
    print(f"[Warn] Weights not found at {weights_path}. Running with random weights.")
    
# Load tuned threshold
threshold = load_threshold(default=0.5, path="weights/threshold.txt")
print(f"[Info] Using decision threshold={threshold:.3f}")

# Store the last 30 states
history = deque(maxlen=30)
# Store timestamps
timestamps = deque(maxlen=30)   

# Short window for smoothing using 5 frames = 25 seconds 
window_probs = deque(maxlen=5)     
window_preds = deque(maxlen=5) 

# UI setup
root = tk.Tk()
root.title("Teacher Dashboard - Student Attention Monitor")
root.geometry("1100x650")

# Video frame
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

video_label = tk.Label(left_frame)
video_label.pack()

status_label = tk.Label(left_frame, text="Status: --", font=("Arial", 14, "bold"))
status_label.pack(pady=(8, 2))

avg_label = tk.Label(left_frame, text="Rolling avg (last 30): --", font=("Arial", 11))
avg_label.pack(pady=(0, 10))

prob_label = tk.Label(left_frame, text="Prob: -- | Smoothed: --", font=("Arial", 10))
prob_label.pack(pady=(0, 10))

# Right side logs and chart
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

log_label = tk.Label(right_frame, text="Attention Log:", font=("Arial", 12, "bold"))
log_label.pack(anchor=tk.W)

log_box = tk.Text(right_frame, height=10, width=60, bg="#f5f5f5", font=("Courier", 10))
log_box.pack(fill=tk.X)

graph_frame = tk.Frame(right_frame)
graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10,0))

# Create a matplotlib figure and axis for plotting
fig, ax = plt.subplots(figsize=(6.5, 3.0))
line, = ax.plot([], [], marker='o')
ax.set_ylim(-0.2, 1.2)
ax.set_ylabel("Attention (1=Yes, 0=No)")
ax.set_xlabel("Time (HH:MM:SS)")
ax.set_title("Student Attention Over Time")

# Embed matplotlib canvas into the Tkinter frame
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Webcam setup with default webcam
cap = cv2.VideoCapture(0)

# Method to update the UI every 5 seconds
def update():
    # Capture from the webcam
    ret, frame = cap.read()
    if not ret:
        # Retry after delay if the frame is not captured
        log_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Camera read failed\n")
        log_box.see(tk.END)
        root.after(1000, update)
        return
    
    # Model input & inference
    model_input = to_model_input_from_bgr(frame)
    # Get raw probability (not thresholded)
    prob = float(model.forward(model_input).squeeze())
    raw_pred = 1 if prob > threshold else 0
    
    # Update smoothing windows
    window_probs.append(prob)
    window_preds.append(raw_pred)
    
    # Smoothed probability & prediction
    sm_prob = rolling_mean(window_probs)
    sm_pred = 1 if sm_prob > threshold else 0
    
    # Update UI text
    now = time.strftime("%H:%M:%S")
    status_text = "Attentive ✅" if sm_pred == 1 else "Inattentive ⚠️"
    prob_label.config(text=f"Prob: {prob:.3f} | Smoothed: {sm_prob:.3f}")
    status_label.config(text=f"Status: {status_text}")
    
    # Append to plotted history (smoothed)
    history.append(sm_pred)
    timestamps.append(now)
    
    # Update log
    log_box.insert(tk.END, f"[{now}] {status_text}\n")
    log_box.see(tk.END)
    
    # Update rolling average label (over the plotted history window)
    avg = rolling_mean(history)
    avg_label.config(text=f"Rolling avg (last {len(history)}): {avg:.2f}")
    
    # Update graph (plot smoothed predictions)
    ax.clear()
    ax.plot(list(range(len(history))), list(history), marker='o', linestyle='-')
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("Attention")
    ax.set_title("Student Attention Over Time")
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels(list(timestamps), rotation=45, ha='right')
    canvas.draw()
    
    # Show live video scaled for the UI
    display_frame = cv2.resize(frame, (320, 240))
    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # schedule next capture (every 5s)
    root.after(5000, update)
    
# Start the main loop
update()
root.mainloop()

# Clean up on close
cap.release()
cv2.destroyAllWindows()