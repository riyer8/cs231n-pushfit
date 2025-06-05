import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score  # type: ignore
from tensorflow.keras import layers, models, preprocessing  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# --- Setup results directory ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results_gru_kfold")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
DATA_ROOT = "datasets/json/mediapipe"
LABELS = {"correct": 1, "wrong": 0}
NUM_KEYPOINTS = 33  # MediaPipe keypoints
FEATURE_DIM = NUM_KEYPOINTS * 3  # x, y, z for each keypoint
MAX_SEQ_LEN = 50  # Max number of frames per sequence
EPOCHS = 100
BATCH_SIZE = 16
K_FOLDS = 5


def extract_sequence_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    sequence = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        if len(keypoints) != NUM_KEYPOINTS:
            continue
        flattened = [kp["x"] for kp in keypoints] + [kp["y"] for kp in keypoints] + [kp["z"] for kp in keypoints]
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


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['wrong', 'correct'], yticklabels=['wrong', 'correct'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_loss(history, path_prefix):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_loss.png")
    plt.close()


def breakdown_by_class(y_array):
    unique, counts = np.unique(y_array, return_counts=True)
    d = dict(zip(unique, counts))
    return d.get(1, 0), d.get(0, 0)


def main():
    print("📦 Loading MediaPipe dataset...")
    X_raw, y = load_dataset()
    print(f"✅ Loaded {len(X_raw)} samples.")

    X_padded = pad_sequences(X_raw, maxlen=MAX_SEQ_LEN)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold = 1
    all_val_acc = []
    all_train_acc = []
    total_cm = np.zeros((2, 2), dtype=int)

    for train_idx, val_idx in skf.split(X_padded, y):
        print(f"\n🔁 Fold {fold}")
        fold_dir = os.path.join(RESULTS_DIR, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_val = X_padded[train_idx], X_padded[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_gru_model(input_shape=(MAX_SEQ_LEN, FEATURE_DIM))
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

        y_val_pred_prob = model.predict(X_val).flatten()
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_val, y_val_pred)
        total_cm += cm

        plot_confusion(y_val, y_val_pred, os.path.join(fold_dir, "confusion_matrix.png"))
        plot_loss(history, os.path.join(fold_dir, "training"))

        train_correct, train_wrong = breakdown_by_class(y_train)
        val_correct, val_wrong = breakdown_by_class(y_val)

        with open(os.path.join(fold_dir, "results.txt"), "w") as f:
            f.write(f"Training accuracy: {train_acc:.4f}\n")
            f.write(f"Validation accuracy: {val_acc:.4f}\n\n")
            f.write(f"Training samples: {len(y_train)} (Correct: {train_correct}, Wrong: {train_wrong})\n")
            f.write(f"Validation samples: {len(y_val)} (Correct: {val_correct}, Wrong: {val_wrong})\n")

        print(f"✅ Fold {fold} complete.")
        fold += 1

    # Save overall results
    avg_train = np.mean(all_train_acc)
    avg_val = np.mean(all_val_acc)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("=== Cross-Validation Summary ===\n")
        f.write(f"Average Train Accuracy: {avg_train:.4f}\n")
        f.write(f"Average Val Accuracy:   {avg_val:.4f}\n\n")
        f.write("Aggregated Confusion Matrix:\n")
        f.write(str(total_cm))

    print(f"\n📊 All results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()