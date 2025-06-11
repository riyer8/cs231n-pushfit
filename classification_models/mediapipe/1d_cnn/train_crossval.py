import os
import json
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

DATA_ROOT = "datasets/json/mediapipe"
LABELS = {"correct": 1, "wrong": 0}
SEQ_LEN = 30
NUM_KEYPOINTS = 33
NUM_FEATURES = NUM_KEYPOINTS * 3
EPOCHS = 100
BATCH_SIZE = 16

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_sequence_from_json(json_path, seq_len=SEQ_LEN):
    with open(json_path, "r") as f:
        data = json.load(f)

    sequence = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        flattened = []
        for kp in keypoints:
            flattened.extend([kp["x"], kp["y"], kp["z"]])
        if len(flattened) == NUM_FEATURES:
            sequence.append(flattened)

    if len(sequence) < seq_len:
        padding = [np.zeros(NUM_FEATURES)] * (seq_len - len(sequence))
        sequence.extend(padding)
    else:
        sequence = sequence[:seq_len]

    return np.array(sequence)

def load_dataset():
    X = []
    y = []
    for label in LABELS:
        folder = os.path.join(DATA_ROOT, label)
        for file in os.listdir(folder):
            if file.endswith(".json"):
                path = os.path.join(folder, file)
                seq = extract_sequence_from_json(path)
                if seq is not None:
                    X.append(seq)
                    y.append(LABELS[label])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["wrong", "correct"], yticklabels=["wrong", "correct"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


def plot_training_history(history, filename_prefix):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{filename_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{filename_prefix}_accuracy.png")
    plt.close()


def save_kfold_results(results, filename):
    with open(filename, "w") as f:
        for i in range(len(results["train_acc"])):
            f.write(f"Fold {i + 1}:\n")
            f.write(f"  Train accuracy: {results['train_acc'][i]:.4f}\n")
            f.write(f"  Val accuracy:   {results['val_acc'][i]:.4f}\n")
            f.write(f"  Train samples: Correct={results['train_counts'][i][1]}, Wrong={results['train_counts'][i][0]}\n")
            f.write(f"  Val samples:   Correct={results['val_counts'][i][1]}, Wrong={results['val_counts'][i][0]}\n\n")

        avg_train = np.mean(results["train_acc"])
        avg_val = np.mean(results["val_acc"])

        f.write("=== Overall ===\n")
        f.write(f"Average Train Accuracy: {avg_train:.4f}\n")
        f.write(f"Average Validation Accuracy: {avg_val:.4f}\n")
        f.write(f"Total Validation Loss: {results['total_loss']:.4f}\n")
        f.write("\nConfusion Matrix (Aggregated):\n")
        f.write(str(results["total_cm"]) + "\n")


def k_fold_cross_validation(X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    total_cm = np.zeros((2, 2), dtype=int)
    total_train_acc = []
    total_val_acc = []
    total_train_counts = []
    total_val_counts = []
    total_loss = []

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n📁 Fold {fold}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(input_shape=(SEQ_LEN, NUM_FEATURES))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        total_loss.append(val_loss)

        y_val_pred_prob = model.predict(X_val)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int).flatten()

        cm = confusion_matrix(y_val, y_val_pred)
        total_cm += cm

        train_counts = np.bincount(y_train, minlength=2)
        val_counts = np.bincount(y_val, minlength=2)

        total_train_acc.append(train_acc)
        total_val_acc.append(val_acc)
        total_train_counts.append(train_counts)
        total_val_counts.append(val_counts)

        fold_dir = os.path.join(RESULTS_DIR, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        plot_confusion_matrix(y_val, y_val_pred, os.path.join(fold_dir, "confusion_matrix.png"))
        plot_training_history(history, os.path.join(fold_dir, "training"))

        print(f"✅ Fold {fold} complete.")
        fold += 1

    return {
        "train_acc": total_train_acc,
        "val_acc": total_val_acc,
        "train_counts": total_train_counts,
        "val_counts": total_val_counts,
        "total_cm": total_cm,
        "total_loss": np.mean(total_loss)
    }


def main():
    print("📦 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    results = k_fold_cross_validation(X, y, k=5)
    save_kfold_results(results, os.path.join(RESULTS_DIR, "kfold_results.txt"))

    print(f"🎉 All results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()