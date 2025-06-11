import os
import json
import numpy as np
from classification import predict_single_video
from math import atan2, degrees
from movenet import extract_keypoints_to_json

# keypoint names from movenet
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# joints to calculate
ANGLES_TO_COMPUTE = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
]

VIDEO_PATH = "llm_pipeline/wrong3.mp4"
KEYPOINTS_OUTPUT_PATH = "llm_pipeline/keypoints_wrong3.json"

# angle between these three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return degrees(angle)

# keypoints from json file
def extract_keypoints_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_keypoints = []
    for frame in data:
        keypoints = frame.get("keypoints", [])
        frame_kps = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if i < len(keypoints):
                frame_kps[name] = (keypoints[i]["x"], keypoints[i]["y"])
        all_keypoints.append(frame_kps)
    return all_keypoints

def compute_angles_over_video(all_keypoints):
    angles_summary = {f"{a[0]}-{a[1]}-{a[2]}": [] for a in ANGLES_TO_COMPUTE}

    for frame in all_keypoints:
        for a, b, c in ANGLES_TO_COMPUTE:
            if a in frame and b in frame and c in frame:
                angle = calculate_angle(frame[a], frame[b], frame[c])
                angles_summary[f"{a}-{b}-{c}"].append(angle)

    averaged = {
        joint: round(np.mean(values), 2) if values else None
        for joint, values in angles_summary.items()
    }
    return averaged

def write_analysis_to_txt(angles, output_txt_path):
    with open(output_txt_path, "w") as f:
        f.write("⚠️ Push-up technique analysis (Incorrect form detected)\n\n")
        for joint, angle in angles.items():
            if angle is not None:
                f.write(f"{joint} angle: {angle} degrees\n")
            else:
                f.write(f"{joint} angle: Not available (missing keypoints)\n")
    print(f"📁 Saved analysis to {output_txt_path}")

def analyze_video_pose(json_path, output_txt_path):
    label = predict_single_video(json_path)

    if label == "wrong":
        print("🚨 Incorrect push-up detected. Extracting details...")
        keypoints = extract_keypoints_from_json(json_path)
        angles = compute_angles_over_video(keypoints)
        write_analysis_to_txt(angles, output_txt_path)
    else:
        print("✅ Push-up form is correct. No further analysis needed.")
    
    return output_txt_path

if __name__ == "__main__":
    JSON_PATH = extract_keypoints_to_json(VIDEO_PATH, KEYPOINTS_OUTPUT_PATH)
    OUTPUT_TXT = "llm_pipeline/feedback_wrong3.txt"
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    output_txt_path = analyze_video_pose(JSON_PATH, OUTPUT_TXT)
