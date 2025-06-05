import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

# Constants
DATA_ROOT = "datasets/json/mediapipe"
LABELS = {"correct": 1, "wrong": 0}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svm_cv")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

def extract_features_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    features = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        if len(keypoints) < 33:
            keypoints += [{"x": 0, "y": 0, "z": 0}] * (33 - len(keypoints))
        elif len(keypoints) > 33:
            keypoints = keypoints[:33]

        flattened = []
        for kp in keypoints:
            flattened.extend([kp["x"], kp["y"], kp["z"]])
        features.append(flattened)

    return np.mean(features, axis=0) if features else np.zeros(99)

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

def plot_confusion_matrix(cm, fold_idx):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["wrong", "correct"],
                yticklabels=["wrong", "correct"])
    plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_fold_{fold_idx+1}.png")
    plt.savefig(path)
    plt.close()

def main():
    print("📦 Loading MediaPipe dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold_idx+1}/{N_SPLITS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        all_fold_accuracies.append(acc)

        print(f"✅ Fold {fold_idx+1} Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

        cm = confusion_matrix(y_val, y_val_pred)
        plot_confusion_matrix(cm, fold_idx)

        # Save per-fold classification report
        report_path = os.path.join(RESULTS_DIR, f"classification_report_fold_{fold_idx+1}.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

    avg_acc = np.mean(all_fold_accuracies)
    std_acc = np.std(all_fold_accuracies)

    print(f"\n📊 Cross-validation complete.")
    print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

    # Save final summary
    summary_path = os.path.join(RESULTS_DIR, "crossval_results.txt")
    with open(summary_path, "w") as f:
        for i, acc in enumerate(all_fold_accuracies):
            f.write(f"Fold {i+1} Accuracy: {acc:.4f}\n")
        f.write(f"\nAverage Accuracy: {avg_acc:.4f}\n")
        f.write(f"Standard Deviation: {std_acc:.4f}\n")

if __name__ == "__main__":
    main()