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
    
    # Evaluate model on test data
    results = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"Precision     : {results['precision']:.4f}")
    print(f"Recall        : {results['recall']:.4f}")
    print(f"F1 Score      : {results['f1_score']:.4f}")
    
    # Save to the evaluation log
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_log.txt", "w") as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
if __name__ == "__main__":
    main()