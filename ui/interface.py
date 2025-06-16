import tkinter as tk
from PIL import Image, ImageTk
import os
import random
import time

class AttentionUI:
    def __init__(self, master, image_folder):
        self.master = master
        self.master.title("Student Attention Monitor")
        
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.current_index = 0

        # UI Components
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.prediction_label = tk.Label(master, text="Prediction: ", font=('Arial', 16))
        self.prediction_label.pack(pady=10)

        self.score_label = tk.Label(master, text="Class Attention Score: ", font=('Arial', 14))
        self.score_label.pack(pady=10)

        self.attention_history = []

        self.update_image()
        
    def update_image(self):
        if self.current_index >= len(self.image_files):
            self.current_index = 0

        img_path = os.path.join(self.image_folder, self.image_files[self.current_index])
        image = Image.open(img_path).resize((200, 200))
        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Simulate prediction (replace with model later)
        prediction = random.choice(["Attentive", "Inattentive"])
        self.prediction_label.config(text=f"Prediction: {prediction}")

        self.attention_history.append(1 if prediction == "Attentive" else 0)
        if len(self.attention_history) > 4:
            self.attention_history.pop(0)

        avg_attention = sum(self.attention_history) / len(self.attention_history) * 100
        self.score_label.config(text=f"Class Attention Score: {avg_attention:.1f}%")

        self.current_index += 1
        self.master.after(5000, self.update_image)  # update every 5 seconds

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionUI(root, image_folder="data/processed/test/attentive")
    root.mainloop()      