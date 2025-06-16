import numpy as np
from cnn import SimpleCNN, binary_cross_entropy, binary_cross_entropy_derivative

# Load preprocessed training data 
X_train = np.load('data/npy/X_train.npy')
y_train = np.load('data/npy/y_train.npy')

# Prepare data types and reshape for training 
X_train = X_train.astype(np.float32)
y_train = y_train.reshape(-1, 1)

# Initialise model and training parameters
model = SimpleCNN()
# Learning rate for gradient descent
learning_rate = 0.01
# Number of training iterations
epochs = 5
# Processing one sample at a time for now so unused
batch_size = 32

# Train a single image and label
def train_one_sample(x, y_true):
    # Perform forward pass and get model output
    y_pred = model.forward(x)
    
    # Calculate loss using binary cross entropy
    loss = binary_cross_entropy(y_true, y_pred)
    
    # Get derivative of loss w.r.t model output
    model.backward(y_true, learning_rate)
    
    return loss, y_pred

# Main training loop for all epochs
def train_model():
    for epoch in range(epochs):
        # Accumulator for loss 
        total_loss = 0
        # Accumulator for correct predictions
        correct = 0
        
        # Loop through each sample (no batching for now)
        for i in range(len(X_train)):
            # One image
            x = X_train[i]
            # Corresponding label
            y =y_train[i]
            
            # Train on one sample
            loss, y_pred  = train_one_sample(x, y)
            
            # Add to total loss
            total_loss += loss
            # Count as correct if prediction and label agree (binary threshold at 0.5)
            if (y_pred > 0.5 and y == 1) or (y_pred <= 0.5 and y == 0):
                correct += 1
          
        # Calculate and display metrics for the epoch      
        acc = correct / len(X_train)
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Accuracy: {acc:.4f}")

# Entry point of the script
if __name__ == "__main__":
    train_model()