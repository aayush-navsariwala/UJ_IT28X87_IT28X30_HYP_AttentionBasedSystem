Student Attention Monitoring (Honours Project)
A lightweight AI system that estimates student attentiveness from still images every 5 seconds and visualises results for a teacher-facing dashboard.
All core model logic (CNN forward/backward, training loop, preprocessing) is implemented from scratch with NumPy. No external libraries are used for grayscaling or resizing (done manually).

Features
-Custom NumPy CNN for binary classification (attentive vs inattentive).
-Image preprocessing pipeline: manual grayscale + nearest-neighbour resize to 48×48.
-Teacher dashboard (Tkinter): live camera feed, timestamped labels, and real-time attention graph.
-Training logs written to CSV for research reporting.

Requirements
-Python 3.9+
-NumPy (core math)
-matplotlib (image I/O in preprocessing & UI graph)
-OpenCV (webcam capture only)
-tkinter (standard library GUI)
-tqdm (progress bar for training)
-Pillow (UI conversion to Tk image only; no processing)

Train the Model
Edit hyperparameters in model/train.py if desired (e.g., subset_size, batch_size, epochs), then:
python model/train.py
Training progress prints per epoch.

Run the Teacher-Facing UI
Shows live webcam feed, timestamped predictions and a live attention graph.
python ui/teacher_dashboard.py
By default, the dashboard samples every 5 seconds. Adjust the interval in teacher_dashboard.py (root.after(5000, update)).

The datasets used in this project were retrieved from: 
https://www.kaggle.com/datasets/msambare/fer2013
https://www.kaggle.com/datasets/tooyoungalex/drivegaze
