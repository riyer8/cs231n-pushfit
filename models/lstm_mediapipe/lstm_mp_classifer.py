import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

# Constants
KEYPOINT_DIM = 3
NUM_KEYPOINTS = 33
SEQ_LENGTH = 100
DATA_DIR = "datasets/json/"
LABELS = {"correct": 1, "wrong": 0}
PLOT_DIR = "results/lstm_mediapipe"
os.makedirs(PLOT_DIR, exist_ok=True)


class PoseSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def load_data():
    all_sequences = []
    all_labels = []
    for label_name, label in LABELS.items():
        folder = os.path.join(DATA_DIR, label_name)
        print(f"Checking folder: {folder}")
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue
        for file_name in os.listdir(folder):
            if file_name.endswith(".json"):
                path = os.path.join(folder, file_name)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    frames = [frame["keypoints"] for frame in data if "keypoints" in frame and len(frame["keypoints"]) == NUM_KEYPOINTS]
                    if len(frames) == 0:
                        print(f"⚠️ No valid frames in: {file_name}")
                        continue

                    seq = np.array([[kp["x"], kp["y"], kp["z"]] for frame in frames for kp in frame])
                    seq = seq.reshape(-1, NUM_KEYPOINTS * KEYPOINT_DIM)

                    if seq.shape[0] < SEQ_LENGTH:
                        pad_len = SEQ_LENGTH - seq.shape[0]
                        pad = np.zeros((pad_len, NUM_KEYPOINTS * KEYPOINT_DIM))
                        seq = np.vstack([seq, pad])
                    else:
                        seq = seq[:SEQ_LENGTH]

                    all_sequences.append(seq)
                    all_labels.append(label)
                except Exception as e:
                    print(f"❌ Failed to load {file_name}: {e}")
    return np.array(all_sequences), np.array(all_labels)


class PoseLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=NUM_KEYPOINTS * KEYPOINT_DIM, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


def plot_training_curves(train_losses, val_accuracies, epochs, suffix=""):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    if val_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"training_curve{suffix}.png")
    plt.savefig(path)
    print(f"📊 Saved training curve to {path}")
    plt.close()


def plot_confusion_matrix(cm, labels, suffix=""):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, f"confusion_matrix{suffix}.png")
    plt.savefig(path)
    print(f"📊 Saved confusion matrix to {path}")
    plt.close()


def train_model(model, train_loader, val_loader=None, epochs=10, suffix=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total
            val_accuracies.append(val_acc)
            print(f"Validation Accuracy: {val_acc:.4f}")

    plot_training_curves(train_losses, val_accuracies, epochs, suffix)
    return model, train_losses, val_accuracies


def main():
    sequences, labels = load_data()
    print(f"Loaded {len(sequences)} samples.")

    k_folds = 5
    epochs = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    all_train_losses = []
    all_val_accuracies = []
    total_cm = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels), 1):
        print(f"\n--- Fold {fold} ---")
        x_train, y_train = sequences[train_idx], labels[train_idx]
        x_val, y_val = sequences[val_idx], labels[val_idx]

        train_dataset = PoseSequenceDataset(x_train, y_train)
        val_dataset = PoseSequenceDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        model = PoseLSTMClassifier()
        model, train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, epochs=epochs, suffix=f"_fold{fold}"
        )
        all_train_losses.append(train_losses)
        all_val_accuracies.append(val_accuracies)

        model.eval()
        all_preds = []
        all_labels_fold = []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels_fold.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels_fold, all_preds)
        total_cm += cm
        plot_confusion_matrix(cm, LABELS.keys(), suffix=f"_fold{fold}")

        acc = accuracy_score(all_labels_fold, all_preds)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)

    print(f"\nAverage validation accuracy over {k_folds} folds: {np.mean(fold_accuracies):.4f}")

    # === Total training curve ===
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_val_accuracies = np.mean(all_val_accuracies, axis=0)
    plot_training_curves(avg_train_losses.tolist(), avg_val_accuracies.tolist(), epochs, suffix="_total")

    # === Total confusion matrix ===
    plot_confusion_matrix(total_cm, LABELS.keys(), suffix="_total")

    # === Final model training on all data ===
    print("\nTraining final model on full dataset...")
    full_dataset = PoseSequenceDataset(sequences, labels)
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    final_model = PoseLSTMClassifier()
    final_model, _, _ = train_model(final_model, full_loader, val_loader=None, epochs=epochs, suffix="_final")

    torch.save(final_model.state_dict(), "pose_lstm_model_final.pt")
    print("✅ Final model saved as pose_lstm_model_final.pt")


if __name__ == "__main__":
    main()