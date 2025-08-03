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

# Model and tracking setup
model = SimpleCNN()
# Store the last 30 states
history = deque(maxlen=30)
timestamps = deque(maxlen=30)

# UI setup
root = tk.Tk()
root.title("Teacher Dashboard - Student Attention Monitor")
root.geometry("1000x600")

# Video frame
video_label = tk.Label(root)
video_label.pack(side=tk.LEFT, padx=10, pady=10)

# Logging frame
log_frame = tk.Frame(root)
log_frame.pack(side=tk.TOP, anchor=tk.NW)

log_label = tk.Label(log_frame, text="Attention Log:", font=("Arial", 12, "bold"))
log_label.pack(anchor=tk.W)

log_box = tk.Text(log_frame, height=20, width=40, bg="#f5f5f5", font=("Courier", 10))
log_box.pack()

# Graph frame
graph_frame = tk.Frame(root)
graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(5, 3))
line, = ax.plot([], [], marker='o')
ax.set_ylim(-0.2, 1.2)
ax.set_ylabel("Attention (1=Yes, 0=No)")
ax.set_xlabel("Time (HH:MM:SS)")
ax.set_title("Student Attention Over Time")

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Webcam setup
cap = cv2.VideoCapture(0)

def update():
    ret, frame = cap.read()
    if not ret:
        root.after(1000, update)
        return
    
    frame = cv2.resize(frame, (200, 200))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Grayscaling and preprocess
    gray = rgb_to_grayscale(rgb)
    resized = resize_image(gray, (48, 48))
    img = resized / 255.0
    
    # Prediction model
    pred = model.predict(img)
    status = "Attentive ✅" if pred == 1 else "Inattentive ⚠️"
    timestamp = time.strftime("%H:%M:%S")
    
    # Update the history and UI
    history.append(pred)
    timestamps.append(timestamp)
    log_box.insert(tk.END, f"[{timestamp}] {status}\n")
    log_box.see(tk.END)
    
    # Update the graph
    ax.clear()
    ax.plot(timestamps, history, marker='o', color='green' if pred else 'red')
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("Attention")
    ax.set_xticklabels(timestamps, rotation=45, ha='right')
    ax.set_title("Student Attention Over Time")
    canvas.draw()
    
    # Display the video in the UI
    img_pil = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # Repeat every 5 seconds
    root.after(5000, update)
    
# Start the loop
update()
root.mainloop()

# Clean up
cap.release()
cv2.destroyAllWindows()