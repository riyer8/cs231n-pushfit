# preprocessing dataset of correct and incorrect sequences

import os
import json
import numpy as np
import torch

DATASET_DIR = "datasets/lstm_classifications"
SEQUENCE_LENGTH = 50
KEYPOINT_DIM = 17 * 2

def load_video_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)
    frames = sorted(data.keys())
    keypoints = []
    for frame in frames:
        coords = data[frame][:34]
        keypoints.append(coords)
    keypoints = np.array(keypoints)
    min_vals = keypoints.min(axis=0)
    max_vals = keypoints.max(axis=0)
    normalized = (keypoints - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized

def normality(seq, length):
    if len(seq) >= length:
        return seq[:length]
    padding = np.zeros((length - len(seq), seq.shape[1]))
    return np.vstack([seq, padding])

def load_dataset():
    X, y = [], []
    for label, folder in enumerate(["correct_sequences", "incorrect_sequences"]):
        folder_path = os.path.join(DATASET_DIR, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                keypoints = load_video_keypoints(os.path.join(folder_path, file))
                keypoints = normality(keypoints, SEQUENCE_LENGTH)
                X.append(keypoints)
                y.append(label)
    X = np.stack(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    X, y = load_dataset()
    print("Dataset loaded:", X.shape, y.shape)