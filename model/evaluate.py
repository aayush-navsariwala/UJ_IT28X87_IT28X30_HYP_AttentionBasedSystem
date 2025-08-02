import numpy as np
from cnn import SimpleCNN
import os

def main():
    # Load preprocessed test data
    X_test = np.load('data/npy/X_test.npy')
    y_test = np.load('data/npy/y_test.npy')
    
    # Ensure proper data shape
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32).reshape(-1)
    
    # Initialise the model 
    model = SimpleCNN()
    
    # Adding weight loading here later
    
    # Evaluate model on test data
    accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save to the evaluation log
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_log.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    
if __name__ == "__main__":
    main()