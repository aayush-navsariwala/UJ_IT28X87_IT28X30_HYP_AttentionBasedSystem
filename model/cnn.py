import os
import numpy as np

# Rectified linear unit activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLu to be used during backpropagation
def relu_derivative(x):
    return (x > 0.0).astype(np.float32)

# Sigmoid activation function to be used in output layer for binary classification
def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

# Derivative of sigmoid to be used during backpropagation
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# Binary cross enthropy loss for binary classification
def binary_cross_entropy(y_true, y_pred, w_pos=1.0, w_neg=1.0):
    eps = 1e-8
    # Stability
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(w_pos * y_true * np.log(y_pred) +
                    w_neg * (1.0 - y_true) * np.log(1.0 - y_pred))

# Derivative of binary cross enthropy loss
def binary_cross_entropy_derivative(y_true, y_pred, w_pos=1.0, w_neg=1.0):
    eps = 1e-8
    # Stability
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    # dL/dŷ for weighted BCE
    return (-(w_pos * y_true) / y_pred) + (w_neg * (1.0 - y_true)) / (1.0 - y_pred)

def maxpool2x2(a: np.ndarray) -> np.ndarray:
    # True max pooling on a 2D feature map
    # Input: a of shape (H, W) with even H and W
    # Output: pooled of shape (H/2, W/2) 
    H, W = a.shape
    out = np.zeros((H // 2, W // 2), dtype=np.float32)
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            out[i // 2, j // 2] = np.max(a[i:i+2, j:j+2])
    return out

# CNN class definition
class SimpleCNN:
    def __init__(self):
        # Set seed for reproducibility of random weight initialisation
        np.random.seed(42)
        
        # Layer 1: Convolutional layer with 1 filter of size 3x3
        conv_fan_in = 9.0
        self.conv1_filter = (np.random.randn(3, 3).astype(np.float32)) * np.sqrt(2.0 / conv_fan_in)
        # Bias term for the convolutional output
        self.conv1_bias   = np.float32(0.0)
        
        # Layer 2: Fully connected dense hidden layer with input size determined by flattened pooled feature map
        # 23x23 pooled output flattened to 529
        fc_fan_in = 23 * 23
        self.fc_weights = (np.random.randn(529, 128).astype(np.float32)) * np.sqrt(2.0 / fc_fan_in)
        # Bias for the 128 neurons in the fully connected hidden layer
        self.fc_bias    = np.zeros((1, 128), dtype=np.float32)
        
        # Layer 3: Output layer with binary classification of 1 output neuron
        self.out_weights = (np.random.randn(128, 1).astype(np.float32)) * np.sqrt(1.0 / 128.0)
        # Bias for the single output neuron
        self.out_bias    = np.zeros((1, 1), dtype=np.float32)
        
    def forward(self, X):
        # Layer 1: Convolutional layer
        # Convolution by applying 3x3 filter across input image (48x48)
        X = X.astype(np.float32, copy=False)
        conv_vals = np.empty((46, 46), dtype=np.float32)
        for i in range(46):
            ii = i + 3
            for j in range(46):
                jj = j + 3
                patch = X[i:ii, j:jj]
                conv_vals[i, j] = np.sum(self.conv1_filter * patch, dtype=np.float32) + self.conv1_bias
        
        # Layer 2: ReLu activation
        # Applying ReLu to have non-linearity
        self.relu1 = relu(conv_vals)
        
        # Layer 3: Max Pooling layer
        # With stride 2 by slicing feature map from 46x46 to 23x23 via true max pooling
        pooled = maxpool2x2(self.relu1)  

        # Layer 4: Flatten layer
        # Flattening the pooled 2D feature map into a 1D vector
        self.fc_input = pooled.reshape(1, -1).astype(np.float32)
        
        # Layer 5: Fully Connected layer
        # Dense layer with ReLu activation (1, 529) to (1, 128)
        self.fc_z = np.dot(self.fc_input, self.fc_weights) + self.fc_bias  
        self.fc_a = relu(self.fc_z)  
        
        # Layer 6: Output layer
        # Dense layer with sigmoid activation for binary classification (1, 128) to (1, 1)
        self.out_z = np.dot(self.fc_a, self.out_weights) + self.out_bias  
        self.out_a = sigmoid(self.out_z)
        
        # Final output with a probability value between 0 and 1
        return self.out_a
    
    def backward(self, y_true, learning_rate, w_pos=1.0, w_neg=1.0):
        # Ensure the array
        if not isinstance(y_true, np.ndarray):
            y_true = np.array([y_true], dtype=np.float32)
        y_true = y_true.reshape(1,).astype(np.float32)
        # Backpropagation through the output layer
        # Compute the derivative of binary cross entropy loss compared to the predicted output
        dLoss_dOut = binary_cross_entropy_derivative(y_true, self.out_a, w_pos=w_pos, w_neg=w_neg)
        dLoss_dOut = dLoss_dOut.reshape(1, 1).astype(np.float32)
        # Compute the derivative of sigmoid activation
        dOut_dZ = sigmoid_derivative(self.out_z).astype(np.float32)
        # Chain rule
        dZ = dLoss_dOut * dOut_dZ  
        # Compute gradients for output weights and biases
        dW_out = np.dot(self.fc_a.T, dZ)
        db_out = dZ

        # Backpropagation through the fully connected layer
        # Backpropagate to fully connected layer activation
        dA_fc = np.dot(dZ, self.out_weights.T)
        # Derivative of ReLu applied to pre-activation
        dZ_fc = dA_fc * relu_derivative(self.fc_z)

        # Gradients for fully connected weights and biases
        dW_fc = np.dot(self.fc_input.T, dZ_fc)
        db_fc = dZ_fc 

        lr = np.float32(learning_rate)
        self.out_weights -= lr * dW_out
        self.out_bias    -= lr * db_out
        self.fc_weights  -= lr * dW_fc
        self.fc_bias     -= lr * db_fc
    
    def save_weights(self, path):
        # Save the model parameters to a single .npz file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 conv1_filter=self.conv1_filter.astype(np.float32),
                 conv1_bias=np.array([self.conv1_bias], dtype=np.float32),
                 fc_weights=self.fc_weights.astype(np.float32),
                 fc_bias=self.fc_bias.astype(np.float32),
                 out_weights=self.out_weights.astype(np.float32),
                 out_bias=self.out_bias.astype(np.float32))
        
    def load_weights(self, path):
        # Load model parameters from a .npz file saved by save_weights().
        data = np.load(path)
        self.conv1_filter = data["conv1_filter"].astype(np.float32)
        self.conv1_bias   = float(data["conv1_bias"][0])
        self.fc_weights   = data["fc_weights"].astype(np.float32)
        self.fc_bias      = data["fc_bias"].astype(np.float32)
        self.out_weights  = data["out_weights"].astype(np.float32)
        self.out_bias     = data["out_bias"].astype(np.float32)
        
    def predict(self, X, threshold=0.5):
        # Performs a binary classification on a given image
        # Perform forward pass to get scalar probability
        prob = float(self.forward(X).squeeze())
        # Applies binary threshold
        return 1 if prob > threshold else 0
    
    # Evaluates the model's accuracy on a given dataset
    def evaluate(self, X, y, threshold=0.5):
        # Return accuracy, precision, recall and F1 score
        TP = FP = FN = TN = 0
        for i in range(len(X)):
            pred = self.predict(X[i], threshold=threshold)
            true = int(np.asarray(y[i]).reshape(-1)[0])  # robust scalar
            if pred == 1 and true == 1:
                TP += 1
            elif pred == 1 and true == 0:
                FP += 1
            elif pred == 0 and true == 1:
                FN += 1
            else:
                TN += 1
        total = TP + TN + FP + FN
        
        # Compute metrics
        accuracy  = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}