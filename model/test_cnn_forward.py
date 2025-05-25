import numpy as np
from cnn import SimpleCNN

# Simulate a 48x48 grayscale image
test_image = np.random.rand(48, 48)

# Instantiate created CNN model
cnn = SimpleCNN()

# Forward pass through CNN
output = cnn.forward(test_image)

# Returns the probability for the attentive class
print("Prediction (probability):", output)

