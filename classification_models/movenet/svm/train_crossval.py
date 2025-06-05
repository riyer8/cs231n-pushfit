import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

# Constants
DATA_ROOT = "datasets/json/movenet"
LABELS = {"correct": 1, "wrong": 0}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_cv_svm_movenet")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

def extract_features_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    features = []

    for frame in data:
        keypoints = frame.get("keypoints", [])
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


def plot_confusion_matrix(cm, fold_idx):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["wrong", "correct"],
                yticklabels=["wrong", "correct"])
    plt.title(f"Confusion Matrix - Fold {fold_idx + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_fold_{fold_idx + 1}.png")
    plt.savefig(cm_path)
    plt.close()


def main():
    print("📦 Loading MoveNet dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} samples.")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_accuracies = []
    val_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold_idx + 1}/{N_SPLITS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"✅ Train Accuracy: {train_acc:.4f}")
        print(f"✅ Val Accuracy:   {val_acc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

        # Save confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        plot_confusion_matrix(cm, fold_idx)

        # Save classification report
        report_path = os.path.join(RESULTS_DIR, f"classification_report_fold_{fold_idx + 1}.txt")
        with open(report_path, "w") as f:
            f.write(f"Train Accuracy: {train_acc:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
            f.write(classification_report(y_val, y_val_pred, target_names=["wrong", "correct"]))

    # Summary
    avg_train = np.mean(train_accuracies)
    avg_val = np.mean(val_accuracies)
    std_val = np.std(val_accuracies)

    print("\n📊 Cross-validation Summary:")
    print(f"Average Train Accuracy: {avg_train:.4f}")
    print(f"Average Val Accuracy:   {avg_val:.4f} ± {std_val:.4f}")

    # Accuracy plot
    plt.figure()
    plt.plot(range(1, N_SPLITS + 1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, N_SPLITS + 1), val_accuracies, label="Val Accuracy", marker='o')
    plt.title("Cross-Validation Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(range(1, N_SPLITS + 1))
    plt.legend()
    acc_plot_path = os.path.join(RESULTS_DIR, "accuracy_plot.png")
    plt.tight_layout()
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"📁 Saved accuracy plot to {acc_plot_path}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "crossval_summary.txt")
    with open(summary_path, "w") as f:
        for i in range(N_SPLITS):
            f.write(f"Fold {i + 1} Train Accuracy: {train_accuracies[i]:.4f}\n")
            f.write(f"Fold {i + 1} Val Accuracy:   {val_accuracies[i]:.4f}\n\n")
        f.write(f"Average Train Accuracy: {avg_train:.4f}\n")
        f.write(f"Average Val Accuracy:   {avg_val:.4f}\n")
        f.write(f"Validation Std Dev:     {std_val:.4f}\n")

    print(f"📁 Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()