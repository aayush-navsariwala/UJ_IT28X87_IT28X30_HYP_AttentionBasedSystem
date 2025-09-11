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
        conv_fan_in = 3 * 3
        self.conv1_filter = (np.random.randn(3, 3).astype(np.float32)) * np.sqrt(2.0 / conv_fan_in)
        # Bias term for the convolutional output
        self.conv1_bias   = np.float32(0.0)
        
        # Layer 2: Fully connected dense hidden layer with input size determined by flattened pooled feature map
        # 23x23 pooled output flattened to 529
        fc_fan_in = 23 * 23
        self.fc_weights = (np.random.randn(529, 128).astype(np.float32)) * np.sqrt(2.0 / fc_fan_in)
        # Bias for the 128 neurons in the fully connected hidden layer
        self.fc_bias = np.zeros((1, 128), dtype=np.float32)
        
        # Layer 3: Output layer with binary classification of 1 output neuron
        self.out_weights = (np.random.randn(128, 1).astype(np.float32)) * np.sqrt(1.0 / 128.0)
        # Bias for the single output neuron
        self.out_bias    = np.zeros((1, 1), dtype=np.float32)
        
        # Caches for backward
        self.X = None
        self.conv_pre = None        
        self.relu1 = None           
        self.pool_mask = None       
        self.pooled = None          
        self.fc_input = None        
        self.fc_z = None
        self.fc_a = None
        self.out_z = None
        self.out_a = None
        
    def _conv_forward_valid_3x3(self, X):
        H, W = X.shape
        out = np.empty((H - 2, W - 2), dtype=np.float32)
        for i in range(H - 2):
            ii = i + 3
            for j in range(W - 2):
                jj = j + 3
                patch = X[i:ii, j:jj]
                out[i, j] = np.sum(self.conv1_filter * patch, dtype=np.float32) + self.conv1_bias
        return out
    
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
                mask_block = (block == m)
                mask[i:i+2, j:j+2] = mask_block
        return pooled, mask
    
    def _maxpool2x2_backward(self, dPooled, mask):
        H2, W2 = dPooled.shape
        dA = np.zeros((H2 * 2, W2 * 2), dtype=np.float32)
        for i in range(H2):
            for j in range(W2):
                block_mask = mask[2*i:2*i+2, 2*j:2*j+2]
                num_true = np.count_nonzero(block_mask)
                if num_true == 0:
                    continue
                dA[2*i:2*i+2, 2*j:2*j+2] += (dPooled[i, j] / num_true) * block_mask.astype(np.float32)
        return dA
        
    def forward(self, X):
        # X: (48,48) float32 in [0,1]
        X = X.astype(np.float32, copy=False)
        self.X = X
        
        # Conv -> ReLU
        self.conv_pre = self._conv_forward_valid_3x3(X)     
        self.relu1 = relu(self.conv_pre)                    

        # MaxPool2x2 (stride 2) -> cache mask
        self.pooled, self.pool_mask = self._maxpool2x2_forward_with_mask(self.relu1)  

        # Flatten
        self.fc_input = self.pooled.reshape(1, -1).astype(np.float32) 

        # FC -> ReLU
        self.fc_z = np.dot(self.fc_input, self.fc_weights) + self.fc_bias   
        self.fc_a = relu(self.fc_z)                                        

        # Output -> Sigmoid
        self.out_z = np.dot(self.fc_a, self.out_weights) + self.out_bias    
        self.out_a = sigmoid(self.out_z)                                    
        return self.out_a

    def backward(self, y_true, learning_rate, w_pos=1.0, w_neg=1.0):
        # y_true -> (1,)
        if not isinstance(y_true, np.ndarray):
            y_true = np.array([y_true], dtype=np.float32)
        y_true = y_true.reshape(1,).astype(np.float32)

        # ----- Output layer grads -----
        # dL/dŷ
        dL_dy = binary_cross_entropy_derivative(y_true, self.out_a, w_pos=w_pos, w_neg=w_neg).reshape(1, 1).astype(np.float32)
        # dŷ/dz
        dy_dz = sigmoid_derivative(self.out_z).astype(np.float32)
        dZ_out = dL_dy * dy_dz                                           

        dW_out = np.dot(self.fc_a.T, dZ_out)                            
        db_out = dZ_out                                                  

        # ----- Backprop into FC hidden -----
        dA_fc = np.dot(dZ_out, self.out_weights.T)                       
        dZ_fc = dA_fc * relu_derivative(self.fc_z)                       

        dW_fc = np.dot(self.fc_input.T, dZ_fc)                           
        db_fc = dZ_fc                                                    

        # ----- Backprop to pooled (flatten -> 23x23) -----
        d_fc_input = np.dot(dZ_fc, self.fc_weights.T)                    
        d_pooled = d_fc_input.reshape(23, 23)                          

        # ----- MaxPool backward -> d_relu1 (46,46) -----
        d_relu1 = self._maxpool2x2_backward(d_pooled, self.pool_mask)    

        # ----- ReLU backward on conv_pre -----
        d_conv_pre = d_relu1 * relu_derivative(self.conv_pre)            

        # ----- Conv weight & bias grads -----
        dW_conv = np.zeros_like(self.conv1_filter, dtype=np.float32)    
        db_conv = np.sum(d_conv_pre).astype(np.float32)                 

        # Each output location (i,j) used input patch X[i:i+3, j:j+3]
        # Accumulate gradient: dW += patch * d_conv_pre[i,j]
        for i in range(46):
            ii = i + 3
            for j in range(46):
                jj = j + 3
                patch = self.X[i:ii, j:jj]                              
                dW_conv += (patch * d_conv_pre[i, j]).astype(np.float32)

        # ----- SGD update -----
        lr = np.float32(learning_rate)
        self.out_weights -= lr * dW_out
        self.out_bias    -= lr * db_out
        self.fc_weights  -= lr * dW_fc
        self.fc_bias     -= lr * db_fc
        self.conv1_filter -= lr * dW_conv
        self.conv1_bias   -= lr * db_conv
    
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