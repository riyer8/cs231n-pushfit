import os
import json
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

# Constants
DATA_ROOT = "datasets/json/mediapipe"
LABELS = {"correct": 1, "wrong": 0}
SEQ_LEN = 30       # number of frames per sample
NUM_KEYPOINTS = 33
NUM_FEATURES = NUM_KEYPOINTS * 3  # x, y, z
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
    # Pad or truncate sequence to seq_len
    if len(sequence) < seq_len:
        # Pad with zeros
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
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["wrong", "correct"], yticklabels=["wrong", "correct"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def plot_training_history(history, filename_prefix):
    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{filename_prefix}_loss.png")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{filename_prefix}_accuracy.png")
    plt.close()

def save_results_txt(train_counts, val_counts, train_acc, val_acc, total_acc, filename):
    with open(filename, "w") as f:
        f.write("Training samples:\n")
        f.write(f"  Correct: {train_counts[1]}\n")
        f.write(f"  Wrong: {train_counts[0]}\n\n")

        f.write("Validation samples:\n")
        f.write(f"  Correct: {val_counts[1]}\n")
        f.write(f"  Wrong: {val_counts[0]}\n\n")

        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Total accuracy: {total_acc:.4f}\n")

def main():
    print("📦 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    # Split train/val (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = build_model(input_shape=(SEQ_LEN, NUM_FEATURES))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2
    )

    print("✅ Training complete.")

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

    # Predict for confusion matrix
    y_val_pred_prob = model.predict(X_val)
    y_val_pred = (y_val_pred_prob > 0.5).astype(int).flatten()

    print("📊 Classification Report (Validation):")
    print(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

    # Plot confusion matrix
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_val, y_val_pred, cm_path)

    # Plot training history
    plot_training_history(history, os.path.join(RESULTS_DIR, "training"))

    # Counts by class in train/val
    train_counts = np.bincount(y_train, minlength=2)
    val_counts = np.bincount(y_val, minlength=2)

    # Total accuracy on combined set
    y_all_pred_prob = model.predict(X)
    y_all_pred = (y_all_pred_prob > 0.5).astype(int).flatten()
    total_acc = accuracy_score(y, y_all_pred)

    # Save results text file
    txt_path = os.path.join(RESULTS_DIR, "results.txt")
    save_results_txt(train_counts, val_counts, train_acc, val_acc, total_acc, txt_path)

    print(f"🎉 Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()