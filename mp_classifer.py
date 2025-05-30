# need to edit

import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
KEYPOINT_DIM = 3  # x, y, z
NUM_KEYPOINTS = 33
SEQ_LENGTH = 100  # Number of frames per sample
DATA_DIR = "mp_pose_outputs"
LABELS = {"correct": 1, "wrong": 0}

# Dataset class
class PoseSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Load and preprocess data
def load_data():
    all_sequences = []
    all_labels = []
    for label_name, label in LABELS.items():
        folder = os.path.join(DATA_DIR, label_name)
        for file_name in os.listdir(folder):
            if file_name.endswith(".json"):
                with open(os.path.join(folder, file_name)) as f:
                    data = json.load(f)
                    frames = [frame["keypoints"] for frame in data if "keypoints" in frame]
                    if len(frames) == 0:
                        continue
                    seq = np.array([[kp["x"], kp["y"], kp["z"]] for frame in frames for kp in frame])
                    seq = seq.reshape(-1, NUM_KEYPOINTS * KEYPOINT_DIM)
                    if len(seq) > SEQ_LENGTH:
                        seq = seq[:SEQ_LENGTH]
                    else:
                        pad_len = SEQ_LENGTH - len(seq)
                        seq = np.pad(seq, ((0, pad_len), (0, 0)))
                    all_sequences.append(seq)
                    all_labels.append(label)
    return np.array(all_sequences), np.array(all_labels)


# Simple LSTM classifier
class PoseLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=NUM_KEYPOINTS * KEYPOINT_DIM, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


# Training loop
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
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
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"Validation Accuracy: {correct / total:.2f}")


# Evaluate model
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS.keys(), yticklabels=LABELS.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Run everything
sequences, labels = load_data()
x_train, x_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, stratify=labels)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

train_dataset = PoseSequenceDataset(x_train, y_train)
val_dataset = PoseSequenceDataset(x_val, y_val)
test_dataset = PoseSequenceDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = PoseLSTMClassifier()
train_model(model, train_loader, val_loader, epochs=10)
evaluate_model(model, test_loader)