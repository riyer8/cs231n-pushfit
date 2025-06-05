import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras import layers, models, preprocessing # type: ignore

# --- Setup results directory ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
DATA_ROOT = "datasets/json/movenet"
LABELS = {"correct": 1, "wrong": 0}
NUM_KEYPOINTS = 17  # MoveNet keypoints count
FEATURE_DIM = NUM_KEYPOINTS * 2  # x, y for each keypoint
MAX_SEQ_LEN = 50  # Max number of frames per sample
EPOCHS = 100
BATCH_SIZE = 16

def extract_sequence_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    sequence = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        if len(keypoints) != NUM_KEYPOINTS:
            # Skip frames with unexpected keypoint counts
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
    # Pad sequences to maxlen with zeros
    padded = preprocessing.sequence.pad_sequences(
        X, maxlen=maxlen, dtype='float32', padding='post', truncating='post', value=0.0
    )
    return padded

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

def main():
    print("📦 Loading MoveNet dataset...")
    X_raw, y = load_dataset()
    print(f"✅ Loaded {len(X_raw)} samples.")

    # Pad sequences
    X = pad_sequences(X_raw, maxlen=MAX_SEQ_LEN)
    y = np.array(y)

    # Split into train and val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

    # Build model
    model = build_lstm_model(input_shape=(MAX_SEQ_LEN, FEATURE_DIM))

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # Predict validation
    y_val_pred_prob = model.predict(X_val).flatten()
    y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

    # Metrics
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    total_acc = accuracy_score(y_val, y_val_pred)

    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Total accuracy on validation set: {total_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['wrong', 'correct'], yticklabels=['wrong', 'correct'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    # Breakdown of training/validation samples by class
    def breakdown_by_class(y_array):
        unique, counts = np.unique(y_array, return_counts=True)
        d = dict(zip(unique, counts))
        return d.get(1, 0), d.get(0, 0)  # correct, wrong

    train_correct, train_wrong = breakdown_by_class(y_train)
    val_correct, val_wrong = breakdown_by_class(y_val)

    # Write results
    results_txt = os.path.join(RESULTS_DIR, "results.txt")
    with open(results_txt, "w") as f:
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Total accuracy (val set): {total_acc:.4f}\n\n")
        f.write(f"Training samples: {len(y_train)} (Correct: {train_correct}, Wrong: {train_wrong})\n")
        f.write(f"Validation samples: {len(y_val)} (Correct: {val_correct}, Wrong: {val_wrong})\n")

    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
