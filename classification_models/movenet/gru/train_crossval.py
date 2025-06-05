import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from tensorflow.keras import layers, models, preprocessing  # type: ignore

# --- Setup results directory ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
DATA_ROOT = "datasets/json/movenet"
LABELS = {"correct": 1, "wrong": 0}
NUM_KEYPOINTS = 17
FEATURE_DIM = NUM_KEYPOINTS * 2
MAX_SEQ_LEN = 50
EPOCHS = 30
BATCH_SIZE = 16
N_SPLITS = 5

def extract_sequence_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    sequence = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        if len(keypoints) != NUM_KEYPOINTS:
            continue
        flattened = []
        for kp in keypoints:
            flattened.extend([kp["x"], kp["y"]])
        sequence.append(flattened)
    return sequence

def load_dataset():
    X, y = [], []
    for label in LABELS:
        folder = os.path.join(DATA_ROOT, label)
        for file in os.listdir(folder):
            if file.endswith(".json"):
                seq = extract_sequence_from_json(os.path.join(folder, file))
                if seq:
                    X.append(seq)
                    y.append(LABELS[label])
    return X, y

def pad_sequences(X, maxlen=MAX_SEQ_LEN):
    return preprocessing.sequence.pad_sequences(
        X, maxlen=maxlen, dtype='float32', padding='post', truncating='post', value=0.0
    )

def build_gru_model(input_shape):
    model = models.Sequential([
        layers.Masking(mask_value=0., input_shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.GRU(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def breakdown_by_class(y_array):
    unique, counts = np.unique(y_array, return_counts=True)
    d = dict(zip(unique, counts))
    return d.get(1, 0), d.get(0, 0)

def main():
    print("📦 Loading MoveNet dataset...")
    X_raw, y = load_dataset()
    print(f"✅ Loaded {len(X_raw)} samples.")

    X = pad_sequences(X_raw, maxlen=MAX_SEQ_LEN)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{N_SPLITS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_gru_model(input_shape=(MAX_SEQ_LEN, FEATURE_DIM))

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=0
        )

        y_val_pred_prob = model.predict(X_val).flatten()
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        total_acc = accuracy_score(y_val, y_val_pred)

        print(f"  ➤ Training accuracy:   {train_acc:.4f}")
        print(f"  ➤ Validation accuracy: {val_acc:.4f}")
        print(f"  ➤ Fold accuracy:       {total_acc:.4f}")

        # Save loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} - Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"fold_{fold}_loss_curve.png"))
        plt.close()

        # Save confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['wrong', 'correct'], yticklabels=['wrong', 'correct'])
        plt.title(f'Fold {fold} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"fold_{fold}_confusion_matrix.png"))
        plt.close()

        train_correct, train_wrong = breakdown_by_class(y_train)
        val_correct, val_wrong = breakdown_by_class(y_val)

        fold_results.append({
            "fold": fold,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "total_acc": total_acc,
            "train_correct": train_correct,
            "train_wrong": train_wrong,
            "val_correct": val_correct,
            "val_wrong": val_wrong
        })

    # Save summary
    results_txt = os.path.join(RESULTS_DIR, "results.txt")
    with open(results_txt, "w") as f:
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Training accuracy:   {result['train_acc']:.4f}\n")
            f.write(f"  Validation accuracy: {result['val_acc']:.4f}\n")
            f.write(f"  Total accuracy:      {result['total_acc']:.4f}\n")
            f.write(f"  Training samples:    {result['train_correct'] + result['train_wrong']} (Correct: {result['train_correct']}, Wrong: {result['train_wrong']})\n")
            f.write(f"  Validation samples:  {result['val_correct'] + result['val_wrong']} (Correct: {result['val_correct']}, Wrong: {result['val_wrong']})\n\n")

        avg_val_acc = np.mean([r["val_acc"] for r in fold_results])
        f.write(f"Average Validation Accuracy over {N_SPLITS} folds: {avg_val_acc:.4f}\n")

    print(f"\n✅ All results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()