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

from utils.image_utils import to_model_input_from_bgr
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

def rolling_mean(deq):
    return float(np.mean(deq)) if len(deq) else 0.0

# Initialising the model
model = SimpleCNN()

# Prefer your existing weights path; keep it configurable
weights_path = "weights/best.npz"  
weights_loaded = False
if os.path.exists(weights_path):
    try:
        model.load(weights_path)            
        print(f"[OK] Loaded weights via model.load() from {weights_path}")
        weights_loaded = True
    except AttributeError:
        if hasattr(model, "load_weights"):
            model.load_weights(weights_path) 
            print(f"[OK] Loaded weights via model.load_weights() from {weights_path}")
            weights_loaded = True
        else:
            print(f"[Warn] Model has no load/load_weights method. Running with random weights.")
else:
    print(f"[Warn] Weights not found at {weights_path}. Running with random weights.")

# Print quick weight sanity regardless (helps confirm load)
try:
    w_mean = float(model.fc_weights.mean())
    w_std  = float(model.fc_weights.std())
    print(f"[DEBUG] fc_weights mean={w_mean:.6f}, std={w_std:.6f}")
except Exception as e:
    print(f"[DEBUG] Could not read fc_weights stats: {e}")

# Tuned decision threshold
threshold = load_threshold(default=0.5, path="weights/threshold.txt")
print(f"[Info] Using decision threshold={threshold:.3f}")

# Store the last 30 states
history = deque(maxlen=30)
timestamps = deque(maxlen=30)   

# Short window for smoothing using 5 frames = 25 seconds 
window_probs = deque(maxlen=1)     
window_preds = deque(maxlen=1) 

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

USE_FACE_CROP = True
face_cascade = None
if USE_FACE_CROP:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[Info] Haar cascade loaded: {cascade_path}")
    else:
        print("[Warn] Haar cascade not found. Continuing without face crop.")
        USE_FACE_CROP = False

def get_face_roi(bgr_frame):
    """Return (roi, found_face) where roi is BGR crop if found, else the full frame."""
    if face_cascade is None:
        return bgr_frame, False
    gray_for_det = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_for_det, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return bgr_frame, False
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.15 * max(w, h))
    H, W = bgr_frame.shape[:2]
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    return bgr_frame[y1:y2, x1:x2], True

# Method to update the UI every 5 seconds
def update():
    ret, frame = cap.read()
    now = time.strftime("%H:%M:%S")

    if not ret:
        log_box.insert(tk.END, f"[{now}] Camera read failed\n")
        log_box.see(tk.END)
        root.after(1000, update)
        return

    # Optional face crop 
    if USE_FACE_CROP:
        frame_roi, found_face = get_face_roi(frame)
    else:
        frame_roi, found_face = frame, True  

    # Preprocess to training-identical tensor (48x48, float32, [0,1])
    x = to_model_input_from_bgr(frame_roi)
    x_min, x_max, x_mean = float(x.min()), float(x.max()), float(x.mean())
    
    # Basic lighting validity 
    valid_lighting = (x_max - x_min) > 0.02

    # Inference 
    y_prob = float(model.forward(x).squeeze())

    # Try to get pre-sigmoid (if your model stores it, as in earlier code)
    y_logit = None
    if hasattr(model, "out_z"):
        try:
            y_logit = float(np.array(model.out_z).squeeze())
        except Exception:
            y_logit = None

    # RAW decision at current threshold
    current_thr = threshold
    raw_pred = 1 if y_prob > current_thr else 0
    
    # Smoothing: update ONLY if we have a valid face ROI + lighting
    if found_face and valid_lighting:
        window_probs.append(y_prob)
        window_preds.append(raw_pred)

    # If window is empty (e.g., first tick or invalid frames so far), fall back to raw prob
    sm_prob = rolling_mean(window_probs) if len(window_probs) else y_prob
    sm_pred = 1 if sm_prob > current_thr else 0

    # UI: show both RAW and SMOOTHED
    raw_text = "Attentive ✅" if raw_pred == 1 else "Inattentive ⚠️"
    sm_text  = "Attentive ✅" if sm_pred == 1 else "Inattentive ⚠️"
    status_label.config(text=f"RAW: {raw_text} | SMOOTHED: {sm_text}")
    prob_label.config(text=f"Prob: {y_prob:.3f} | Smoothed: {sm_prob:.3f} | Thr: {current_thr:.2f}")
    
    # Debug log (face flag, ROI size, validity, weights, logit)
    roi_h, roi_w = frame_roi.shape[:2]
    dbg_face = "Y" if found_face else "N"
    dbg_valid = "Y" if valid_lighting else "N"
    dbg_logit = f"{y_logit:.3f}" if y_logit is not None else "N/A"
    log_box.insert(
        tk.END,
        (f"[{now}] RAW={raw_text}, SMOOTHED={sm_text} | "
         f"prob={y_prob:.3f} (thr={current_thr:.2f}) | "
         f"x[min/mean/max]={x_min:.3f}/{x_mean:.3f}/{x_max:.3f} | "
         f"face={dbg_face} roi={roi_w}x{roi_h} valid={dbg_valid} | "
         f"weights_loaded={weights_loaded} | logit={dbg_logit}\n")
    )
    log_box.see(tk.END)
    
    # Rolling average label (over plotted history window)
    avg = rolling_mean(history)
    avg_label.config(text=f"Rolling avg (last {len(history)}): {avg:.2f}")

    # Redraw chart
    ax.clear()
    ax.plot(list(range(len(history))), list(history), marker='o', linestyle='-')
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("Attention")
    ax.set_title("Student Attention Over Time")
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels(list(timestamps), rotation=45, ha='right')
    canvas.draw()

    # Show live video preview (draw face box if used)
    display = frame.copy()
    if USE_FACE_CROP and face_cascade is not None:
        gray_for_det = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_for_det, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            pad = int(0.15 * max(w, h))
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    display_frame = cv2.resize(display, (360, 270))
    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule next capture (every 5s)
    root.after(5000, update)

# Start loop
update()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()