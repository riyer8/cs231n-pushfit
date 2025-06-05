import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

# --- Setup results directory ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results_cv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
DATA_ROOT = "datasets/json/movenet"
LABELS = {"correct": 1, "wrong": 0}
NUM_KEYPOINTS = 17
FEATURE_DIM = NUM_KEYPOINTS * 2  # x and y for each keypoint
EPOCHS = 100
BATCH_SIZE = 16
N_SPLITS = 5

def extract_features_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    features = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        flattened = []
        for kp in keypoints:
            flattened.extend([kp["x"], kp["y"]])
        features.append(flattened)
    return np.mean(features, axis=0) if features else np.zeros(FEATURE_DIM)

def load_dataset():
    X, y = [], []
    for label in LABELS:
        folder = os.path.join(DATA_ROOT, label)
        for file in os.listdir(folder):
            if file.endswith(".json"):
                path = os.path.join(folder, file)
                features = extract_features_from_json(path)
                if features is not None:
                    X.append(features)
                    y.append(LABELS[label])
    return np.array(X), np.array(y)

def build_1d_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_loss_curve(history, fold_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss - Fold {fold_idx+1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_curve_fold_{fold_idx+1}.png"))
    plt.close()

def plot_confusion_matrix(cm, fold_idx):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['wrong', 'correct'],
                yticklabels=['wrong', 'correct'])
    plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_fold_{fold_idx+1}.png"))
    plt.close()

def main():
    print("📦 Loading MoveNet dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    X = X[..., np.newaxis]  # shape: (samples, timesteps=features, channels=1)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold_idx+1}/{N_SPLITS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_1d_cnn(input_shape=X_train.shape[1:])

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=0
        )

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"✅ Training Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"📉 Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        y_val_pred_prob = model.predict(X_val).flatten()
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_val, y_val_pred)
        plot_confusion_matrix(cm, fold_idx)
        plot_loss_curve(history, fold_idx)

    avg_train_acc = np.mean(train_accuracies)
    avg_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    results_txt = os.path.join(RESULTS_DIR, "crossval_results.txt")
    with open(results_txt, "w") as f:
        for i in range(N_SPLITS):
            f.write(f"Fold {i+1} Training Accuracy: {train_accuracies[i]:.4f} | Loss: {train_losses[i]:.4f}\n")
            f.write(f"Fold {i+1} Validation Accuracy: {val_accuracies[i]:.4f} | Loss: {val_losses[i]:.4f}\n\n")
        f.write(f"Average Training Accuracy: {avg_train_acc:.4f}\n")
        f.write(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}\n")
        f.write(f"Average Training Loss: {avg_train_loss:.4f}\n")
        f.write(f"Average Validation Loss: {avg_val_loss:.4f}\n")

    print("\n📊 Cross-validation complete.")
    print(f"📈 Avg Training Acc: {avg_train_acc:.4f}")
    print(f"📈 Avg Validation Acc: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"📁 Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()