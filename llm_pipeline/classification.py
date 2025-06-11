import joblib # type: ignore
import json
import numpy as np
import os
from movenet import extract_keypoints_to_json

# static model path
MODEL_PATH = "classification_models/movenet/random_forest/results/random_forest_model.joblib"
VIDEO_PATH = "llm_pipeline/wrong3.mp4"
KEYPOINTS_OUTPUT_PATH = "llm_pipeline/keypoints_wrong3.json"

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

def predict_single_video(json_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train it first.")

    clf = joblib.load(MODEL_PATH)
    features = extract_features_from_json(json_path).reshape(1, -1)
    prediction = clf.predict(features)[0]
    label = "correct" if prediction == 1 else "wrong"
    print(f"🧠 Prediction: {label}")
    return label

if __name__ == "__main__":
    JSON_PATH = extract_keypoints_to_json(VIDEO_PATH, KEYPOINTS_OUTPUT_PATH)
    predict_single_video(JSON_PATH)
