import csv
import os
import numpy as np
from tqdm import tqdm
from cnn import SimpleCNN, binary_cross_entropy 

# Configuration for training
np.random.seed(42)
subset_size = 5000            
learning_rate = 0.001         
epochs = 3
batch_size = 32
val_ratio = 0.10              
early_stopping_patience = 2
weights_path = "weights/best.npz"
log_path = "logs/training_log.csv"

# Balance the subsets and stratified split
def build_balanced_subset(X, y, subset_size):
    # Return a balanced subset (half positive and half negative)
    y_flat = y.reshape(-1).astype(int)
    idx_pos = np.where(y_flat == 1)[0]
    idx_neg = np.where(y_flat == 0)[0]
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)
    
    half = subset_size // 2
    take_pos = min(half, len(idx_pos))
    take_neg = min(subset_size - take_pos, len(idx_neg))
    sel = np.concatenate([idx_pos[:take_pos], idx_neg[:take_neg]])
    np.random.shuffle(sel)

    Xb = np.ascontiguousarray(X[sel].astype(np.float32))
    yb = y[sel].reshape(-1, 1)
    return Xb, yb

def stratified_split(Xb, yb, val_ratio=0.10):
    #Stratified train/val split
    y_flat = yb.reshape(-1).astype(int)
    pos_idx = np.where(y_flat == 1)[0]
    neg_idx = np.where(y_flat == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    val_pos = max(1, int(len(pos_idx) * val_ratio))
    val_neg = max(1, int(len(neg_idx) * val_ratio))
    
    val_sel = np.concatenate([pos_idx[:val_pos], neg_idx[:val_neg]])
    tr_sel  = np.concatenate([pos_idx[val_pos:], neg_idx[val_neg:]])
    np.random.shuffle(val_sel)
    np.random.shuffle(tr_sel)
    return (np.ascontiguousarray(Xb[tr_sel]),
            yb[tr_sel],
            np.ascontiguousarray(Xb[val_sel]),
            yb[val_sel])

# Load preprocessed training data 
X_train = np.load('data/npy/X_train.npy').astype(np.float32)
y_train = np.load('data/npy/y_train.npy')

# Guard to ensure inputs are [0,1]
if X_train.max() > 1.5:
    X_train /= 255.0

# Try to load a true validation split from disk; fall back to in-memory split
val_path_X = 'data/npy/X_val.npy'
val_path_y = 'data/npy/y_val.npy'
has_on_disk_val = os.path.exists(val_path_X) and os.path.exists(val_path_y)

if has_on_disk_val:
    X_val = np.load(val_path_X).astype(np.float32)
    y_val = np.load(val_path_y)
    if X_val.max() > 1.5:
        X_val /= 255.0
        
    # Build the balanced subset only from the training split
    X_tr_base, y_tr_base = X_train, y_train
    X_tr, y_tr = build_balanced_subset(X_tr_base, y_tr_base, subset_size=subset_size)
else:
    # No on-disk val → build balanced subset from training pool, then internal stratified split
    Xb, yb = build_balanced_subset(X_train, y_train, subset_size=subset_size)
    X_tr, y_tr, X_val, y_val = stratified_split(Xb, yb, val_ratio=val_ratio)

# Sanity print class counts
print(f"Train counts -> pos:{int(np.sum(y_tr))} neg:{len(y_tr) - int(np.sum(y_tr))}")
print(f"Val counts -> pos:{int(np.sum(y_val))} neg:{len(y_val) - int(np.sum(y_val))}")

# Class weights computed on train only
pos = int(np.sum(y_tr))
neg = int(len(y_tr) - pos)
w_pos = (neg / max(1, pos))  
w_neg = 1.0
print(f"class weights -> w_pos:{w_pos:.3f} w_neg:{w_neg:.3f}")

# Initialise model and training parameters
model = SimpleCNN()

# Train a single batch of images and update their weights
def train_one_batch(batch_x, batch_y):
    batch_loss = 0.0
    correct = 0
    for i in range(len(batch_x)):
        x = batch_x[i].reshape(48, 48)
        y_scalar = int(np.asarray(batch_y[i]).reshape(-1)[0])
        y_vec = np.array([y_scalar], dtype=np.float32)

        y_pred = model.forward(x)  # (1,1)
        loss = binary_cross_entropy(y_vec, y_pred, w_pos=w_pos, w_neg=w_neg)
        model.backward(y_vec, learning_rate, w_pos=w_pos, w_neg=w_neg)

        batch_loss += float(loss)
        prob = float(y_pred.squeeze())
        if (prob > 0.5 and y_scalar == 1) or (prob <= 0.5 and y_scalar == 0):
            correct += 1
    return batch_loss / len(batch_x), correct

def eval_one_epoch(Xd, yd):
    total_loss = 0.0
    total_correct = 0
    for i in range(0, len(Xd), batch_size):
        end_i = min(i + batch_size, len(Xd))
        bx, by = Xd[i:end_i], yd[i:end_i]
        for j in range(len(bx)):
            x = bx[j].reshape(48, 48)
            y_scalar = int(np.asarray(by[j]).reshape(-1)[0])
            y_vec = np.array([y_scalar], dtype=np.float32)

            y_pred = model.forward(x)
            total_loss += float(binary_cross_entropy(y_vec, y_pred, w_pos=w_pos, w_neg=w_neg))
            prob = float(y_pred.squeeze())
            if (prob > 0.5 and y_scalar == 1) or (prob <= 0.5 and y_scalar == 0):
                total_correct += 1
    avg_loss = total_loss / len(Xd)
    acc = total_correct / len(Xd)
    return avg_loss, acc

def find_best_threshold(Xv, yv, thresholds=np.linspace(0.05, 0.95, 37)):
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        TP=FP=TN=FN=0
        for i in range(len(Xv)):
            x = Xv[i].reshape(48, 48)
            y_true = int(np.asarray(yv[i]).reshape(-1)[0])
            pred = model.predict(x, threshold=t)
            if y_true==1 and pred==1: TP+=1
            elif y_true==0 and pred==1: FP+=1
            elif y_true==0 and pred==0: TN+=1
            elif y_true==1 and pred==0: FN+=1
        precision = TP/(TP+FP) if TP+FP>0 else 0.0
        recall    = TP/(TP+FN) if TP+FN>0 else 0.0
        f1 = (2*precision*recall)/(precision+recall) if precision+recall>0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1
        
# Main training loop for all epochs
def train_model():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    best_val_acc = 0.0
    patience = 0

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAccuracy", "ValLoss", "ValAcc"])

        for epoch in range(1, epochs + 1):
            # Shuffle train split each epoch
            idx = np.arange(len(X_tr))
            np.random.shuffle(idx)
            Xs, ys = X_tr[idx], y_tr[idx]

            train_loss_sum = 0.0
            train_correct_sum = 0

            for i in tqdm(range(0, len(Xs), batch_size), desc=f"Epoch {epoch}/{epochs}"):
                end_i = min(i + batch_size, len(Xs))
                bx, by = Xs[i:end_i], ys[i:end_i]
                b_loss, b_correct = train_one_batch(bx, by)
                train_loss_sum += b_loss * len(bx)
                train_correct_sum += b_correct

            train_loss = train_loss_sum / len(Xs)
            train_acc  = train_correct_sum / len(Xs)

            val_loss, val_acc = eval_one_epoch(X_val, y_val)

            print(f"Epoch {epoch} — "
                  f"Train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
                  f"Val: loss {val_loss:.4f}, acc {val_acc:.4f}")

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            # Early stopping + checkpoint + threshold tuning on VAL
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                model.save_weights(weights_path)

                best_t, best_f1 = find_best_threshold(X_val, y_val)
                with open("weights/threshold.txt", "w") as tf:
                    tf.write(str(best_t))
                print(f"✅ Saved new best to {weights_path} (val_acc={best_val_acc:.4f}); "
                      f"chosen threshold={best_t:.3f} (val F1={best_f1:.3f})")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("⏹️ Early stopping triggered.")
                    break

# Entry point of the script
if __name__ == "__main__":
    train_model()