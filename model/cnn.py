import os
import numpy as np

# Activation functions with their derivatives
# Rectified linear unit activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLu to be used during backpropagation 
def relu_derivative(x):
    return (x > 0.0).astype(np.float32)

# Sigmoid activation function to be used in output layer for binary classification
def sigmoid(x):
    # Clip to avoid exp overflow for numeric stability
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

# Derivative of sigmoid to be used during backpropagation
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# Weighted binary cross enthropy loss for binary classification
def binary_cross_entropy(y_true, y_pred, w_pos=1.0, w_neg=1.0):
    eps = 1e-8
    # Numeric stability
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

# True max pooling with stride 2 on a single 2D feature map
def maxpool2x2(a: np.ndarray) -> np.ndarray:
    H, W = a.shape
    out = np.zeros((H // 2, W // 2), dtype=np.float32)
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            out[i // 2, j // 2] = np.max(a[i:i+2, j:j+2])
    return out

# CNN class definition
class SimpleCNN:
    """
    CNN for 48x48 grayscale images:

      Input (48x48)
        - Conv(3x3, valid) -> (46x46)
        - ReLU
        - MaxPool(2x2, stride 2) -> (23x23)
        - Flatten -> 529
        - Dense(529 -> 128) + ReLU
        - Dense(128 -> 1) + Sigmoid (probability)
        
    """
    
    def __init__(self):
        # Set seed for reproducibility of random weight initialisation
        np.random.seed(42)
        
        # Layer 1: Convolutional layer with single filter of size 3x3
        conv_fan_in = 3 * 3
        self.conv1_filter = (np.random.randn(3, 3).astype(np.float32)) * np.sqrt(2.0 / conv_fan_in)
        # Scalar bias for the convolution output feature map.
        self.conv1_bias = np.float32(0.0)
        
        # Layer 2: Fully connected hidden layer 
        # 48x48 -> 46x46 -> MaxPool 2x2 -> 23x23 flattened -> 529 features
        fc_fan_in = 23 * 23
        self.fc_weights = (np.random.randn(529, 128).astype(np.float32)) * np.sqrt(2.0 / fc_fan_in)
        # Bias for the 128 neurons in the fully connected hidden layer
        self.fc_bias = np.zeros((1, 128), dtype=np.float32)
        
        # Layer 3: Output layer with binary classification of 1 output neuron
        self.out_weights = (np.random.randn(128, 1).astype(np.float32)) * np.sqrt(1.0 / 128.0)
        self.out_bias = np.zeros((1, 1), dtype=np.float32)
        
        # Storing immediate tensors produced in forward() to reuse in backprop
        self.X = None               # Input image of 48x48
        self.conv_pre = None        # Pre-activation convolution output of 46x46 
        self.relu1 = None           # After ReLU function added of 46x46
        self.pool_mask = None       # Mask for MaxPool positions of 46x46 boolean
        self.pooled = None          # After MaxPool of 23x23
        self.fc_input = None        # Flattened pool of 1x529
        self.fc_z = None            # Pre-activation dense hidden of 1x128
        self.fc_a = None            # Post-activation dense hidden of 1x128
        self.out_z = None           # Pre-activation output of 1x1
        self.out_a = None           # Sigmoid output probability of 1x1
        
    # Convolution helper for explicit 3x3 sliding window
    def _conv_forward_valid_3x3(self, X):
        H, W = X.shape
        # Output size is valid for 3x3 kernel
        out = np.empty((H - 2, W - 2), dtype=np.float32)
        for i in range(H - 2):
            # End of index for the 3 row window
            ii = i + 3
            for j in range(W - 2):
                # End of index for the 3 column window
                jj = j + 3
                # 3x3 local receptive field
                patch = X[i:ii, j:jj]
                # Convolution calculation
                out[i, j] = np.sum(self.conv1_filter * patch, dtype=np.float32) + self.conv1_bias
        return out
    
    # Max pool forward pass with mask 
    def _maxpool2x2_forward_with_mask(self, A):
        H, W = A.shape 
        assert H % 2 == 0 and W % 2 == 0
        pooled = np.empty((H // 2, W // 2), dtype=np.float32)
        mask = np.zeros_like(A, dtype=bool)
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                block = A[i:i+2, j:j+2]
                m = np.max(block)
                pooled[i // 2, j // 2] = m
                # True where the max occurred
                mask_block = (block == m)
                mask[i:i+2, j:j+2] = mask_block
        return pooled, mask
    
    # Max pool backward pass
    def _maxpool2x2_backward(self, dPooled, mask):
        H2, W2 = dPooled.shape
        dA = np.zeros((H2 * 2, W2 * 2), dtype=np.float32)
        for i in range(H2):
            for j in range(W2):
                # Check which inputs were maxima
                block_mask = mask[2*i:2*i+2, 2*j:2*j+2]
                num_true = np.count_nonzero(block_mask)
                if num_true == 0:
                    continue
                # Distribute gradient to the max locations
                dA[2*i:2*i+2, 2*j:2*j+2] += (dPooled[i, j] / num_true) * block_mask.astype(np.float32)
        return dA
        
    # Forward pass
    def forward(self, X):
        # Ensure dtype and cache input for backprop
        X = X.astype(np.float32, copy=False)
        self.X = X
        
        # Convolution -> 46x46
        self.conv_pre = self._conv_forward_valid_3x3(X) 
        # ReLU activation    
        self.relu1 = relu(self.conv_pre)                    

        # MaxPool (2x2) -> 23x23 and store mask for backward routing of gradients
        self.pooled, self.pool_mask = self._maxpool2x2_forward_with_mask(self.relu1)  

        # Flatten pooled feature map -> 1x529 
        self.fc_input = self.pooled.reshape(1, -1).astype(np.float32) 

        # Dense hidden (1x529) * (529x128) + bias -> 1x128
        self.fc_z = np.dot(self.fc_input, self.fc_weights) + self.fc_bias
        # ReLU on hidden   
        self.fc_a = relu(self.fc_z)                                        

        # Output (1x128) * (128x1) + bias -> 1x1
        self.out_z = np.dot(self.fc_a, self.out_weights) + self.out_bias    
        # Sigmoid probability in (0,1)
        self.out_a = sigmoid(self.out_z)                                    
        
        return self.out_a

    # Backward pass
    def backward(self, y_true, learning_rate, w_pos=1.0, w_neg=1.0):
        # Ensure y_true is a (1,) float32 array for consistency
        if not isinstance(y_true, np.ndarray):
            y_true = np.array([y_true], dtype=np.float32)
        y_true = y_true.reshape(1,).astype(np.float32)

        # Output layer gradients
        # dL/dy_hat (1x1)
        dL_dy = binary_cross_entropy_derivative(y_true, self.out_a, w_pos=w_pos, w_neg=w_neg).reshape(1, 1).astype(np.float32)
        # dy_hat/dz_out = sigmoid(z_out)
        dy_dz = sigmoid_derivative(self.out_z).astype(np.float32)
        # dL/dz_out
        dZ_out = dL_dy * dy_dz                                           

        # Gradients for output weights/bias
        # dW_out = (fc_a)^T * dZ_out  -> (128x1)
        dW_out = np.dot(self.fc_a.T, dZ_out) 
        # db_out = dZ_out (broadcast over batch=1)                           
        db_out = dZ_out                                                  

        # Backprop to hidden dense layer 
        # dL/da_fc = dZ_out * W_out^T  -> (1x128)
        dA_fc = np.dot(dZ_out, self.out_weights.T)  
        # dL/dz_fc = dL/da_fc ⊙ ReLU'(z_fc)  -> (1x128)                     
        dZ_fc = dA_fc * relu_derivative(self.fc_z)                       

        # Gradients for fc weights/bias:
        # dW_fc = (fc_input)^T * dZ_fc  -> (529x128)
        dW_fc = np.dot(self.fc_input.T, dZ_fc)       
        # 1x128                    
        db_fc = dZ_fc                                                    

        # Backprop to pooled (unflatten)
        # dL/d(fc_input) = dZ_fc * W_fc^T -> (1x529)
        d_fc_input = np.dot(dZ_fc, self.fc_weights.T)    
        # Reshape back to 23x23 to match pooled map                
        d_pooled = d_fc_input.reshape(23, 23)                          

        # MaxPool backward 
        # Route gradients to positions that were maxima -> (46x46)
        d_relu1 = self._maxpool2x2_backward(d_pooled, self.pool_mask)    

        # ReLU backward on conv_pre
        # dL/d(conv_pre) = d_relu1 ⊙ ReLU'(conv_pre)
        d_conv_pre = d_relu1 * relu_derivative(self.conv_pre)            

        # Convolution weight/bias gradients
        # Initialise accumulators for 3x3 kernel and scalar bias.
        dW_conv = np.zeros_like(self.conv1_filter, dtype=np.float32)    
        # Bias gradient is sum of upstream gradients
        db_conv = np.sum(d_conv_pre).astype(np.float32)                 

        # Each output location (i,j) used input patch X[i:i+3, j:j+3]
        # Accumulate gradient: dW += patch * d_conv_pre[i,j]
        for i in range(46):
            ii = i + 3
            for j in range(46):
                jj = j + 3
                # 3x3 slice from input
                patch = self.X[i:ii, j:jj]                              
                dW_conv += (patch * d_conv_pre[i, j]).astype(np.float32)

        # SGD parameter update
        lr = np.float32(learning_rate)
        self.out_weights -= lr * dW_out
        self.out_bias    -= lr * db_out
        self.fc_weights  -= lr * dW_fc
        self.fc_bias     -= lr * db_fc
        self.conv1_filter -= lr * dW_conv
        self.conv1_bias   -= lr * db_conv
    
    # Save and load weights
    def save_weights(self, path):
        # Save the model parameters to a single .npz file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Stored arrays: conv1_filter, conv1_bias, fc_weights, fc_bias, out_weights, out_bias
        np.savez(path,
                 conv1_filter=self.conv1_filter.astype(np.float32),
                 # Keeps bias as (1,) for reload consistency
                 conv1_bias=np.array([self.conv1_bias], dtype=np.float32),
                 fc_weights=self.fc_weights.astype(np.float32),
                 fc_bias=self.fc_bias.astype(np.float32),
                 out_weights=self.out_weights.astype(np.float32),
                 out_bias=self.out_bias.astype(np.float32))
        
    def load_weights(self, path):
        # Load model parameters from a .npz file saved by save_weights().
        data = np.load(path)
        self.conv1_filter = data["conv1_filter"].astype(np.float32)
        # Restore scalar
        self.conv1_bias   = float(data["conv1_bias"][0])
        self.fc_weights   = data["fc_weights"].astype(np.float32)
        self.fc_bias      = data["fc_bias"].astype(np.float32)
        self.out_weights  = data["out_weights"].astype(np.float32)
        self.out_bias     = data["out_bias"].astype(np.float32)
    
    # Inference helpers  
    def predict(self, X, threshold=0.5):
        # Performs a binary classification on a given image
        # Perform forward pass to get scalar probability
        prob = float(self.forward(X).squeeze())
        # Applies binary threshold
        return 1 if prob > threshold else 0
    
    # Evaluates the model's accuracy given datasets
    def evaluate(self, X, y, threshold=0.5):
        # Return accuracy, precision, recall and F1 score
        TP = FP = FN = TN = 0
        for i in range(len(X)):
            pred = self.predict(X[i], threshold=threshold)
            # Robustly coerce label to scalar int to handle shapes like (1,)
            true = int(np.asarray(y[i]).reshape(-1)[0])
            if pred == 1 and true == 1:   TP += 1
            elif pred == 1 and true == 0: FP += 1
            elif pred == 0 and true == 1: FN += 1
            else:                         TN += 1
        total = TP + TN + FP + FN
        
        # Compute metrics
        accuracy  = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}