import os
import numpy as np

# Rectified linear unit activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLu to be used during backpropagation
def relu_derivative(x):
    return (x > 0).astype(float)

# Sigmoid activation function to be used in output layer for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid to be used during backpropagation
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Binary cross enthropy loss for binary classification
def binary_cross_entropy(y_true, y_pred):
    # Small value to prevent log(0)
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Derivative of binary cross enthropy loss
def binary_cross_entropy_derivative(y_true, y_pred):
    eps = 1e-8
    return (-(y_true / (y_pred + eps)) + ((1 - y_true) / (1 - y_pred + eps)))

# Converts image to column matrix to simplify convolution as a matrix multiplication and assumes single channel image
def im2col(image, kernel_size=3, stride=1):
    # Get the height and width of the inputted image
    H, W = image.shape
    
    # Calculate the output feature map dimensions after convolution
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    
    # Create a matrix to hold flattened kernel-sized patches
    cols = np.zeros((kernel_size * kernel_size, out_h * out_w))
    
    # Index to track where to insert each flattened patch
    col_index = 0
    
    # Slide the kernel over the image with the given stride
    for i in range(0, H - kernel_size + 1, stride):
        for j in range(0, W - kernel_size + 1, stride):
            # Extract a kernel-sized patch from the image 
            patch = image[i:i+kernel_size, j:j+kernel_size]
            
            # Flatten the patch into a 1D vector and store as a column
            cols[:, col_index] = patch.flatten()
            
            # Move to the next column index
            col_index += 1
    # cols: matrix where each column is a flattened patch
    # out_h, out_w: the output dimensions after convolution        
    return cols, out_h, out_w

# CNN class definition
class SimpleCNN:
    def __init__(self):
        # Set seed for reproducibility of random weight initialisation
        np.random.seed(42)
        
        # Layer 1: Convolutional layer with 1 filter of size 3x3
        self.conv1_filter = np.random.randn(3, 3) * 0.01
        # Bias term for the convolutional output
        self.conv1_bias = 0.0
        
        # Layer 2: Fully connected dense hidden layer with input size determined by flattened pooled feature map
        # 23x23 pooled output flattened to 529
        self.fc_weights = np.random.randn(529, 128) * 0.01
        # Bias for the 128 neurons in the fully connected hidden layer
        self.fc_bias = np.zeros((1, 128))
        
        # Layer 3: Output layer with binary classification of 1 output neuron
        self.out_weights = np.random.randn(128, 1) * 0.01
        # Bias for the single output neuron
        self.out_bias = np.zeros((1, 1))
        
    def forward(self, X):
        # Layer 1: Convolutional layer
        # Convolution by applying 3x3 filter across input image (48x48)
        self.cols = []
        # Vertical sliding window
        for i in range(46):
            # Horizontal sliding window
            for j in range(46):
                # 3x3 region from input
                patch = X[i:i+3, j:j+3]
                val = np.sum(self.conv1_filter * patch) + self.conv1_bias
                self.cols.append(val)
        # Reshape to 2D feature map
        conv_output = np.array(self.cols).reshape(46, 46)
        
        # Layer 2: ReLu activation
        # Applying ReLu to have non-linearity
        self.relu1 = relu(conv_output)
        
        # Layer 3: Max Pooling layer
        # Using 2x2 with stride 2 by slicing feature map from 46x46 to 23x23
        pooled = self.relu1[::2, ::2]
        
        # Layer 4: Flatten layer
        # Flattening the pooled 2D feature map into a 1D vector
        self.flatten = pooled.flatten().reshape(1, -1)
        
        # Layer 5: Fully Connected layer
        # Dense layer with ReLu activation (1, 529) to (1, 128)
        self.fc_input = self.flatten
        self.fc_z = np.dot(self.fc_input, self.fc_weights) + self.fc_bias
        self.fc_a = relu(self.fc_z)
        
        # Layer 6: Output layer
        # Dense layer with sigmoid activation for binary classification (1, 128) to (1, 1)
        self.out_z = np.dot(self.fc_a, self.out_weights) + self.out_bias
        self.out_a = sigmoid(self.out_z)
        
        # Final output with a probability value between 0 and 1
        return self.out_a
    
    def backward(self, y_true, learning_rate):
        # Backpropagation through the output layer
        # Compute the derivative of binary cross entropy loss compared to the predicted output
        dLoss_dOut = binary_cross_entropy_derivative(y_true, self.out_a)
        # Compute the derivative of sigmoid activation
        dOut_dZ = sigmoid_derivative(self.out_z)
        # Chain rule
        dZ = dLoss_dOut * dOut_dZ  
        # Compute gradients for output weights and biases
        dW_out = np.dot(self.fc_a.T, dZ)
        db_out = np.sum(dZ, axis=0, keepdims=True)

        # Backpropagation through the fully connected layer
        # Backpropagate to fully connected layer activation
        dA_fc = np.dot(dZ, self.out_weights.T)
        # Derivative of ReLu applied to pre-activation
        dZ_fc = dA_fc * relu_derivative(self.fc_z)

        # Gradients for fully connected weights and biases
        dW_fc = np.dot(self.fc_input.T, dZ_fc)
        db_fc = np.sum(dZ_fc, axis=0, keepdims=True)

        # Gradient descent update
        # Update the output layer weights and biases
        self.out_weights -= learning_rate * dW_out
        self.out_bias -= learning_rate * db_out
        
        # Update fully connected layer weights and bias
        self.fc_weights -= learning_rate * dW_fc
        self.fc_bias -= learning_rate * db_fc
    
    def save_weights(self, path):
        # Save the model parameters to a single .npz file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            conv1_filter=self.conv1_filter,
            conv1_bias=np.array([self.conv1_bias], dtype=np.float32),
            fc_weights=self.fc_weights,
            fc_bias=self.fc_bias,
            out_weights=self.out_weights,
            out_bias=self.out_bias
        )
        
    def load_weights(self, path):
        
        # Load model parameters from a .npz file saved by save_weights().
        data = np.load(path)
        self.conv1_filter = data["conv1_filter"]
        self.conv1_bias   = float(data["conv1_bias"][0])
        self.fc_weights   = data["fc_weights"]
        self.fc_bias      = data["fc_bias"]
        self.out_weights  = data["out_weights"]
        self.out_bias     = data["out_bias"]
        
    
    def predict(self, X):
        # Performs a binary classification on a given image
        # Perform forward pass to get scalar probability
        prob = float(self.forward(X).squeeze())
        # Applies binary threshold
        return 1 if prob > 0.5 else 0
    
    # Evaluates the model's accuracy on a given dataset
    def evaluate(self, X, y):
        # Return accuracy, precision, recall and F1 score
        TP = FP = FN = TN = 0
        
        for i in range(len(X)):
            # Predict class
            pred = self.predict(X[i])
            true = y[i]
            # Compare with ground truth 
            if pred == 1 and true == 1:
                TP += 1
            elif pred == 1 and true == 0:
                FP += 1
            elif pred == 0 and true == 1:
                FN += 1
            elif pred == 0 and true == 0:
                TN += 1
                
        # Compute metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Return proportion of correct predictions
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }