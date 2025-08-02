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
        # For reproducibility
        np.random.seed(42)
        
        # Layer 1: Convolutional layer with 1 filter of size 3x3
        self.conv1_filter = np.random.randn(3, 3) * 0.01
        self.conv1_bias = 0.0
        
        # Layer 2: Fully connected layer with input size determined by flattened pooled feature map
        # 23x23 pooled output flattened to 529
        self.fc_weights = np.random.randn(529, 128) * 0.01
        self.fc_bias = np.zeros((1, 128))
        
        # Layer 3: Output layer with binary classification of 1 output neuron
        self.out_weights = np.random.randn(128, 1) * 0.01
        self.out_bias = np.zeros((1, 1))
        
    def forward(self, X):
        # Convolution by applying 3x3 filter across input image
        self.cols = []
        for i in range(46):
            for j in range(46):
                patch = X[i:i+3, j:j+3]
                val = np.sum(self.conv1_filter * patch) + self.conv1_bias
                self.cols.append(val)
        conv_output = np.array(self.cols).reshape(46, 46)
        
        # Activation of ReLu
        self.relu1 = relu(conv_output)
        
        # Max pooling using 2x2 with stride 2 by manually downsampling by slicing
        pooled = self.relu1[::2, ::2]
        
        # Flattening the pooled feature map
        self.flatten = pooled.flatten().reshape(1, -1)
        
        # Fully connected layer with FC -> ReLu
        self.fc_input = self.flatten
        self.fc_z = np.dot(self.fc_input, self.fc_weights) + self.fc_bias
        self.fc_a = relu(self.fc_z)
        
        # Output layer with FC -> sigmoid for binary output
        self.out_z = np.dot(self.fc_a, self.out_weights) + self.out_bias
        self.out_a = sigmoid(self.out_z)
        
        return self.out_a
    
    def backward(self, y_true, learning_rate):
        m = y_true.shape[0]

        # Output layer
        dLoss_dOut = binary_cross_entropy_derivative(y_true, self.out_a)
        dOut_dZ = sigmoid_derivative(self.out_z)
        dZ = dLoss_dOut * dOut_dZ  # shape: (1, 1)

        dW_out = np.dot(self.fc_a.T, dZ)
        db_out = np.sum(dZ, axis=0, keepdims=True)

        # Fully connected layer
        dA_fc = np.dot(dZ, self.out_weights.T)
        dZ_fc = dA_fc * relu_derivative(self.fc_z)

        dW_fc = np.dot(self.fc_input.T, dZ_fc)
        db_fc = np.sum(dZ_fc, axis=0, keepdims=True)

        # Gradient descent update
        self.out_weights -= learning_rate * dW_out
        self.out_bias -= learning_rate * db_out
        self.fc_weights -= learning_rate * dW_fc
        self.fc_bias -= learning_rate * db_fc
    
    def predict(self, X):
        # Predicts the binary class for a given image
        output = self.forward(X)
        return 1 if output > 0.5 else 0
    
    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == y[i]:
                correct += 1
        return correct / len(X)