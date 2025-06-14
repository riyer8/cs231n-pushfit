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

def compute_additional_metrics(all_keypoints):
    left_elbow_angles = []
    right_elbow_angles = []
    spine_deviation_list = []
    rep_threshold = 40
    pushup_reps = 0
    rep_state = False

    for frame in all_keypoints:
        if all(k in frame for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            angle = calculate_angle(frame["left_shoulder"], frame["left_elbow"], frame["left_wrist"])
            left_elbow_angles.append(angle)
        if all(k in frame for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            angle = calculate_angle(frame["right_shoulder"], frame["right_elbow"], frame["right_wrist"])
            right_elbow_angles.append(angle)

        if all(k in frame for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            shoulder_center = np.mean([frame["left_shoulder"], frame["right_shoulder"]], axis=0)
            hip_center = np.mean([frame["left_hip"], frame["right_hip"]], axis=0)
            delta = np.array(hip_center) - np.array(shoulder_center)
            angle = abs(degrees(atan2(delta[1], delta[0])))
            spine_deviation_list.append(angle)

        if "right_shoulder" in frame and "right_elbow" in frame and "right_wrist" in frame:
            angle = calculate_angle(frame["right_shoulder"], frame["right_elbow"], frame["right_wrist"])
            if angle < rep_threshold and not rep_state:
                rep_state = True
            elif angle >= rep_threshold and rep_state:
                pushup_reps += 1
                rep_state = False

    metrics = {
        "left_elbow_range": round(max(left_elbow_angles) - min(left_elbow_angles), 2) if left_elbow_angles else None,
        "right_elbow_range": round(max(right_elbow_angles) - min(right_elbow_angles), 2) if right_elbow_angles else None,
        "avg_spine_deviation": round(np.mean(spine_deviation_list), 2) if spine_deviation_list else None,
        "reps_count": pushup_reps
    }
    return metrics

def write_analysis_to_txt(angles, output_txt_path, metrics=None):
    with open(output_txt_path, "w") as f:
        f.write("⚠️ Push-up technique analysis (Incorrect form detected)\n\n")

        f.write("=== Joint Angles (Average) ===\n")
        for joint, angle in angles.items():
            if angle is not None:
                f.write(f"{joint} angle: {angle} degrees\n")
            else:
                f.write(f"{joint} angle: Not available (missing keypoints)\n")

        if metrics:
            f.write("\n=== Summary Metrics ===\n")
            if metrics["left_elbow_range"] is not None:
                f.write(f"Left elbow angle range: {metrics['left_elbow_range']} degrees\n")
            if metrics["right_elbow_range"] is not None:
                f.write(f"Right elbow angle range: {metrics['right_elbow_range']} degrees\n")
            if metrics["avg_spine_deviation"] is not None:
                f.write(f"Average spine deviation: {metrics['avg_spine_deviation']} degrees\n")
            f.write(f"Estimated number of push-up reps: {metrics['reps_count']}\n")

    print(f"📁 Saved analysis to {output_txt_path}")

def analyze_video_pose(json_path, output_txt_path):
    label = predict_single_video(json_path)

    if label == "wrong":
        print("🚨 Incorrect push-up detected. Extracting details...")
        keypoints = extract_keypoints_from_json(json_path)
        angles = compute_angles_over_video(keypoints)
        metrics = compute_additional_metrics(keypoints)
        write_analysis_to_txt(angles, output_txt_path, metrics=metrics)
    else:
        print("✅ Push-up form is correct. No further analysis needed.")
    
    return output_txt_path

def keypoints_to_feedback_path(keypoints_path: str) -> str:
    dir_path, filename = os.path.split(keypoints_path)
    name_without_ext = os.path.splitext(filename)[0]

    if name_without_ext.startswith("keypoints_"):
        base_name = name_without_ext[len("keypoints_"):]
    else:
        base_name = name_without_ext

    feedback_filename = f"feedback_{base_name}.txt"
    return os.path.join(dir_path, feedback_filename)