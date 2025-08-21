import numpy as np
from cnn import SimpleCNN
import os
from tqdm import tqdm

# Set to 5000 for now. Need to change to None to run whole set
MAX_SAMPLES = 5000
WEIGHTS_PATH = "weights/best.npz"
THRESHOLD_PATH = "weights/threshold.txt"

def safe_div(a, b):
    return (a / b) if b != 0 else 0.0

def load_threshold(default=0.5):
    th = default
    try:
        with open(THRESHOLD_PATH, "r") as f:
            th = float(f.read().strip())
            print(f"Using saved threshold: {th:.3f}")
    except FileNotFoundError:
        print(f"No threshold file found at {THRESHOLD_PATH}; using default {default:.2f}")
    except Exception as e:
        print(f"Warning: failed to read threshold file ({e}); using default {default:.2f}")
    return th

def main():
    # 1) Load data
    X_test = np.load('data/npy/X_test.npy').astype(np.float32)
    y_test = np.load('data/npy/y_test.npy').astype(np.int32).reshape(-1)

    # Shuffle the test set first
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

    # Optional subset for speed
    if MAX_SAMPLES is not None:
        pos_idx = np.where(y_test == 1)[0]
        neg_idx = np.where(y_test == 0)[0]
        
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        
        per_class = MAX_SAMPLES // 2
        take_pos = min(per_class, len(pos_idx))
        take_neg = min(per_class, len(neg_idx))
        sel = np.concatenate([pos_idx[:take_pos], neg_idx[:take_neg]])
        rng.shuffle(sel)
        
        X_test = X_test[sel]
        y_test = y_test[sel]
        
    # Report class distribution before evaluating
    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    print(f"Class distribution in evaluation set → positives={n_pos}, negatives={n_neg}")
    if n_pos == 0 or n_neg == 0:
        print("One class is missing; metrics will be misleading for precision/recall/F1.")

    # Init model (and optionally load trained weights)
    model = SimpleCNN()  
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"Weights not found at {WEIGHTS_PATH}. Evaluating with random weights.")

    # Load decision threshold (defaults to 0.5 if file missing)
    threshold = load_threshold(default=0.5)
    
    # Confusion matrix counters
    TP = FP = TN = FN = 0

    # Evaluate with a progress bar
    for i in tqdm(range(len(X_test)), desc="Evaluating"):
        x = X_test[i].reshape(48, 48)
        y_true = int(y_test[i])

        # Use the tuned threshold in predict()
        y_pred = model.predict(x, threshold=threshold)

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
    print(f"Threshold    : {threshold:.3f}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1_score:.4f}")
    print(f"TP={TP} FP={FP} TN={TN} FN={FN}")

    # Save log
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_log.txt", "w") as f:
        f.write(f"samples={len(X_test)}\n")
        f.write(f"threshold={threshold:.6f}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"precision={precision:.6f}\n")
        f.write(f"recall={recall:.6f}\n")
        f.write(f"f1_score={f1_score:.6f}\n")
        f.write(f"TP={TP} FP={FP} TN={TN} FN={FN}\n")

if __name__ == "__main__":
    main()