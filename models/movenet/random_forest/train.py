import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

# Constants
DATA_ROOT = "datasets/json/movenet"
LABELS = {"correct": 1, "wrong": 0}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def extract_features_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    features = []

    for frame in data:
        keypoints = frame.get("keypoints", [])

        # Ensure exactly 17 keypoints
        if len(keypoints) < 17:
            keypoints += [{"x": 0, "y": 0}] * (17 - len(keypoints))
        elif len(keypoints) > 17:
            keypoints = keypoints[:17]

        flattened = []
        for kp in keypoints:
            flattened.extend([kp["x"], kp["y"]])
        features.append(flattened)

    return np.mean(features, axis=0) if features else np.zeros(34)


def load_dataset():
    X = []
    y = []

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


def count_class_distribution(y, label_dict):
    counts = {name: int(np.sum(y == value)) for name, value in label_dict.items()}
    return counts


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("📦 Loading MoveNet dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"📊 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    train_counts = count_class_distribution(y_train, LABELS)
    val_counts = count_class_distribution(y_val, LABELS)

    # Train model
    print("🌲 Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_total_true = np.concatenate([y_train, y_val])
    y_total_pred = np.concatenate([y_train_pred, y_val_pred])

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    total_acc = accuracy_score(y_total_true, y_total_pred)

    print(f"✅ Training Accuracy: {train_acc:.4f}")
    print(f"✅ Validation Accuracy: {val_acc:.4f}")
    print(f"✅ Total Accuracy: {total_acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["wrong", "correct"], yticklabels=["wrong", "correct"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"📁 Saved confusion matrix to {cm_path}")

    # Accuracy plot
    plt.figure()
    plt.plot(["Train", "Validation"], [train_acc, val_acc], marker='o')
    plt.title("Training vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    acc_plot_path = os.path.join(RESULTS_DIR, "accuracy_plot.png")
    plt.tight_layout()
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"📁 Saved accuracy plot to {acc_plot_path}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Total Accuracy: {total_acc:.4f}\n")
        f.write("\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"  Correct: {train_counts['correct']}\n")
        f.write(f"  Wrong: {train_counts['wrong']}\n")
        f.write(f"Validation Samples: {len(X_val)}\n")
        f.write(f"  Correct: {val_counts['correct']}\n")
        f.write(f"  Wrong: {val_counts['wrong']}\n")
        f.write(f"Total Samples: {len(X)}\n")
    print(f"📁 Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()