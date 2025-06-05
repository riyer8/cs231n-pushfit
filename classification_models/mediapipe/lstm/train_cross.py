import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models, preprocessing  # type: ignore

# --- Setup results directory ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results_lstm_cv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
DATA_ROOT = "datasets/json/mediapipe"
LABELS = {"correct": 1, "wrong": 0}
NUM_KEYPOINTS = 33
FEATURE_DIM = NUM_KEYPOINTS * 3
MAX_SEQ_LEN = 50
EPOCHS = 100
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
            flattened.extend([kp["x"], kp["y"], kp["z"]])
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

def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Masking(mask_value=0., input_shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, fold_idx):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['wrong', 'correct'],
                yticklabels=['wrong', 'correct'])
    plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_fold_{fold_idx+1}.png"))
    plt.close()

def plot_loss_curve(history, fold_idx):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss - Fold {fold_idx+1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_curve_fold_{fold_idx+1}.png"))
    plt.close()

def main():
    print("📦 Loading MediaPipe dataset...")
    X_raw, y = load_dataset()
    print(f"✅ Loaded {len(X_raw)} samples.")

    X = pad_sequences(X_raw, maxlen=MAX_SEQ_LEN)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    all_fold_train_accuracies = []
    all_fold_val_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold_idx+1}/{N_SPLITS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_lstm_model(input_shape=(MAX_SEQ_LEN, FEATURE_DIM))

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=0
        )

        # Get final training and validation accuracy from history
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        all_fold_train_accuracies.append(train_acc)
        all_fold_val_accuracies.append(val_acc)

        print(f"✅ Fold {fold_idx+1} Training Accuracy:   {train_acc:.4f}")
        print(f"✅ Fold {fold_idx+1} Validation Accuracy: {val_acc:.4f}")

        # Predict on validation set
        y_val_pred_prob = model.predict(X_val).flatten()
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

        # Confusion Matrix and Loss Curve
        cm = confusion_matrix(y_val, y_val_pred)
        plot_confusion_matrix(cm, fold_idx)
        plot_loss_curve(history, fold_idx)

    avg_train_acc = np.mean(all_fold_train_accuracies)
    std_train_acc = np.std(all_fold_train_accuracies)
    avg_val_acc = np.mean(all_fold_val_accuracies)
    std_val_acc = np.std(all_fold_val_accuracies)

    # Save results to file
    results_txt = os.path.join(RESULTS_DIR, "crossval_results.txt")
    with open(results_txt, "w") as f:
        for i in range(N_SPLITS):
            f.write(f"Fold {i+1} Training Accuracy:   {all_fold_train_accuracies[i]:.4f}\n")
            f.write(f"Fold {i+1} Validation Accuracy: {all_fold_val_accuracies[i]:.4f}\n\n")
        f.write(f"Average Training Accuracy:   {avg_train_acc:.4f} ± {std_train_acc:.4f}\n")
        f.write(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}\n")

    print(f"\n📊 Cross-validation complete.")
    print(f"Average Training Accuracy:   {avg_train_acc:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"📁 Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()