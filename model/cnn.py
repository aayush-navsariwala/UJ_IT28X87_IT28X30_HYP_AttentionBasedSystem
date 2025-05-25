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
    return (-(y_true / (y_pred + eps)) + ((1 - y_true) / (1 - y_pred + eps))) / len(y_true)

# Converts image to column matrix to simplify convolution as a matrix multiplication and assumes single channel image
def im2col(image, kernel_size=3, stride=1):
    H, W = image.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    cols = np.zeros((kernel_size * kernel_size, out_h * out_w))
    
    col_index = 0
    for i in range(0, H - kernel_size + 1, stride):
        for j in range(0, W - kernel_size + 1, stride):
            # Flatten the 3x3 patch and store it 
            patch = image[i:i+kernel_size, j:j+kernel_size].flatten()
            cols[:, col_index] = patch
            col_index += 1
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
        self.cols, h, w = im2col(X, 3)
        conv_out = np.dot(self.conv1_filter.flatten(), self.cols) + self.conv1_bias
        conv_out = conv_out.reshape(h, w)
        
        # Activation of ReLu
        self.relu1 = relu(conv_out)
        
        # Max pooling using 2x2 with stride 2 by manually downsampling by slicing
        pooled = self.relu1[::2, ::2]
        
        # Flattening the pooled feature map
        self.flatten = pooled.flatten().reshape(1, -1)
        
        # Fully connected layer with FC -> ReLu
        self.fc_out = relu(np.dot(self.flatten, self.fc_weights) + self.fc_bias)
        
        # Output layer with FC -> sigmoid for binary output
        self.output_raw = np.dot(self.fc_out, self.out_weights) + self.out_bias
        self.output = sigmoid(self.output_raw)
        
        return self.output
    
    def predict(self, X):
        # Predicts the binary class for a given image
        output = self.forward(X)
        return 1 if output > 0.5 else 0
    
    
    