import csv
import os
import numpy as np
from tqdm import tqdm
from cnn import SimpleCNN, binary_cross_entropy 

# Configuration
np.random.seed(42)
subset_size = 6969
learning_rate = 0.01
epochs = 5
batch_size = 32
val_ratio = 0.10
early_stopping_patience = 3
weights_path = "weights/best.npz"
log_path = "logs/training_log.csv"

# Load preprocessed training data 
X = np.load('data/npy/X_train.npy')
y = np.load('data/npy/y_train.npy')

# Shuffle the dataset
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx][:subset_size].astype(np.float32)
y = y[idx][:subset_size].reshape(-1, 1)

# Train/Val split
n = len(X)
split = int(n * (1 - val_ratio))
X_tr, X_val = X[:split], X[split:]
y_tr, y_val = y[:split], y[split:]

pos = int(np.sum(y_tr))
neg = len(y_tr) - pos
w_pos = (neg / max(1, pos)) 
w_neg = 1.0
print(f"class weights -> w_pos:{w_pos:.3f} w_neg:{w_neg:.3f}")

# Initialise model and training parameters
model = SimpleCNN()

# Train a single batch of images and update their weights
def train_one_batch(batch_x, batch_y):
    # Sum of batch losses
    batch_loss = 0.0
    # Count of correct predictions
    correct = 0
    
    for i in range(len(batch_x)):
        # Ensure image shape is 48x48
        x = batch_x[i].reshape(48, 48)
        # Extract scalar label
        y_scalar = int(batch_y[i])
        # Convert it to a 1D NumPy array with shape
        y_vec = np.array([y_scalar], dtype=np.float32)  
        
        # Forward and backward pass
        y_pred = model.forward(x)
        # Calculate loss
        loss = binary_cross_entropy(y_vec, y_pred, w_pos=w_pos, w_neg=w_neg)
        # Update weights
        model.backward(y_vec, learning_rate, w_pos=w_pos, w_neg=w_neg)
        
        batch_loss += loss
        # Count correct predictions
        if (y_pred > 0.5 and y_scalar == 1) or (y_pred <= 0.5 and y_scalar == 0):
            correct += 1
    
    # Return average loss and correct predictions
    return batch_loss / len(batch_x), correct

def eval_one_epoch(Xd, yd):
    # Evaluate loss/accuracy on a dataset without updating the weights
    total_loss = 0.0
    total_correct = 0
    for i in range(0, len(Xd), batch_size):
        end_i = min(i + batch_size, len(Xd))
        bx = Xd[i:end_i]
        by = yd[i:end_i]
        
        # Micro-batch loop for consistency with predict()
        for j in range(len(bx)):
            x = bx[j].reshape(48, 48)
            y_scalar = int(by[j])
            y_vec = np.array([y_scalar])
            y_pred = model.forward(x)
            total_loss += binary_cross_entropy(y_vec, y_pred, w_pos=w_pos, w_neg=w_neg)
            if (y_pred > 0.5 and y_scalar == 1) or (y_pred <= 0.5 and y_scalar == 0):
                total_correct += 1
    avg_loss = total_loss / len(Xd)
    acc = total_correct / len(Xd)
    return avg_loss, acc

# Main training loop for all epochs
def train_model():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Best accuracy seen so far
    best_val_acc = 0.0
    # Early stopping counter
    patience = 0
    
    # CSV logging
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Row headers
        writer.writerow(["Epoch", "TrainLoss", "TrainAccuracy", "ValLoss", "ValAcc"])

        for epoch in range(1, epochs + 1):
            # Shuffle training data for each epoch
            idx = np.arange(len(X_tr))
            np.random.shuffle(idx)
            Xs = X_tr[idx]
            ys = y_tr[idx]
            
            # Train epoch
            train_loss_sum = 0.0
            train_correct_sum = 0
        
            # Iterate through batces
            for i in tqdm(range(0, len(Xs), batch_size), desc=f"Epoch {epoch}/{epochs}"):
                end_i = min(i + batch_size, len(Xs))
                bx, by = Xs[i:end_i], ys[i:end_i]
                b_loss, b_correct = train_one_batch(bx, by)
                train_loss_sum += b_loss * len(bx)
                train_correct_sum += b_correct
                
            train_loss = train_loss_sum / len(Xs)
            train_acc = train_correct_sum / len(Xs)
            
            # Validate epochs
            val_loss, val_acc = eval_one_epoch(X_val, y_val)
            
            print(f"Epoch {epoch} — "
                  f"Train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
                  f"Val: loss {val_loss:.4f}, acc {val_acc:.4f}")
            
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Reset patience if improved
                patience = 0
                # Saving the improved trained model
                model.save_weights(weights_path)
                print(f"Saved new best weights to {weights_path} (val_acc={best_val_acc:.4f})")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

# Entry point of the script
if __name__ == "__main__":
    train_model()