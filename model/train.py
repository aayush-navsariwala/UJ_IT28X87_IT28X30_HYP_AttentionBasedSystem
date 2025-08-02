import csv
import os
import numpy as np
from cnn import SimpleCNN, binary_cross_entropy, binary_cross_entropy_derivative
from tqdm import tqdm 

# Load preprocessed training data 
X_train = np.load('data/npy/X_train.npy')
y_train = np.load('data/npy/y_train.npy')

# Shuffle the dataset
indices = np.arrange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

# Use only a subset of the data for faster training (limited to 20000)
subset_size = 20000
X_train = X_train[:subset_size].astype(np.float32)
y_train = y_train[:subset_size].reshape(-1, 1)

# Initialise model and training parameters
model = SimpleCNN()

# Step size for gradient descent
learning_rate = 0.01

# Number of training iterations per data
epochs = 10

# Number of samples processed per training step
batch_size = 32

# Stop if accuracy does not improve for 2 epochs
early_stopping_patience = 2

# Train a single batch of images
def train_one_batch(batch_x, batch_y):
    # Sum of batch losses
    batch_loss = 0
    # Count of correct predictions
    correct = 0
    
    for i in range(len(batch_x)):
        # Ensure image shape is 48x48
        x = batch_x[i].reshape(48, 48)
        # Convert label to float
        y = float(batch_y[i])
        
        # Forward and backward pass
        y_pred = model.forward(x)
        # Calculate loss
        loss = binary_cross_entropy(y, y_pred)
        # Update weights
        model.backward(y, learning_rate)
        
        batch_loss += loss
        # Count correct predictions
        if (y_pred > 0.5 and y == 1) or (y_pred <= 0.5 and y == 0):
            correct += 1
    
    # Return average loss and correct predictions
    return batch_loss / len(batch_x), correct

# Main training loop for all epochs
def train_model():
    # Best accuracy seen so far
    best_acc = 0
    # Early stopping counter
    patience = 0
    
    # Create logs folder and CSV file for logging training accuracy
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "training_log.csv")
    
    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        # Row headers
        writer.writerow(["Epoch", "Loss", "Accuracy"])
    
        for epoch in range(epochs):
            # Total loss for the epoch 
            total_loss = 0
            # Total correct predictions
            total_correct = 0
        
            # Shuffle at start of epoch
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
        
            # Iterate through batces
            for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                end_i = min(i + batch_size, len(X_train))
                batch_x = X_train_shuffled[i:end_i]
                batch_y = y_train_shuffled[i:end_i]
            
                # Train on this batch
                batch_loss, correct = train_one_batch(batch_x, batch_y)
                total_loss += batch_loss * len(batch_x)
                total_correct += correct
            
            # Compute and display metrics  
            avg_loss = total_loss / len(X_train)
            acc = total_correct / len(X_train)
            print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} — Accuracy: {acc:.4f}")
            
            # Write to the CSV log
            writer.writerow([epoch + 1, avg_loss, acc])
        
            # Early stopping logic
            if acc > best_acc:
                best_acc = acc
                # Reset patience if improved
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

# Entry point of the script
if __name__ == "__main__":
    train_model()