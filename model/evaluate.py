import numpy as np
from cnn import SimpleCNN
import os
from tqdm import tqdm

# ---------------- Config ----------------
# For a quick run, start with a subset. Set to None to use entire test set.
MAX_SAMPLES = 5000
THRESHOLD = 0.5
WEIGHTS_PATH = "weights/best.npz"
# ----------------------------------------

def safe_div(a, b):
    return (a / b) if b != 0 else 0.0

def main():
    # Load data
    X_test = np.load('data/npy/X_test.npy').astype(np.float32)
    y_test = np.load('data/npy/y_test.npy').astype(np.int32).reshape(-1)

    # Optional subset for speed
    if MAX_SAMPLES is not None:
        X_test = X_test[:MAX_SAMPLES]
        y_test = y_test[:MAX_SAMPLES]

    # Init model (and optionally load trained weights)
    model = SimpleCNN()
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)

    # Confusion matrix counters
    TP = FP = TN = FN = 0

    # Evaluate with a progress bar
    for i in tqdm(range(len(X_test)), desc="Evaluating"):
        x = X_test[i].reshape(48, 48)
        y_true = int(y_test[i])

        # Model outputs 0/1 via predict()
        y_pred = model.predict(x)  # uses 0.5 threshold internally

        # Update confusion counts
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1

    # Metrics
    accuracy  = safe_div(TP + TN, TP + TN + FP + FN)
    precision = safe_div(TP, TP + FP)
    recall    = safe_div(TP, TP + FN)
    f1_score  = safe_div(2 * precision * recall, precision + recall)

    # Print
    print("\nResults")
    print(f"Samples      : {len(X_test)}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1_score:.4f}")
    print(f"TP={TP} FP={FP} TN={TN} FN={FN}")

    # Save log
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_log.txt", "w") as f:
        f.write(f"samples={len(X_test)}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"precision={precision:.6f}\n")
        f.write(f"recall={recall:.6f}\n")
        f.write(f"f1_score={f1_score:.6f}\n")
        f.write(f"TP={TP} FP={FP} TN={TN} FN={FN}\n")

if __name__ == "__main__":
    main()

""""
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
"""