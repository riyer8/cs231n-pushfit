# use mediapipe to extract positions

import cv2 # type: ignore
import mediapipe as mp # type: ignore
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

DATA_ROOT = "datasets/videos"
OUTPUT_ROOT = "datasets/json/mediapipe/"

os.makedirs(os.path.join(OUTPUT_ROOT, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "wrong"), exist_ok=True)

def extract_pose_from_video(video_path, output_json_path, overlay=False):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

        keypoints_all.append({"keypoints": frame_keypoints})

        if overlay and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Pose Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    with open(output_json_path, 'w') as f:
        json.dump(keypoints_all, f, indent=2)

    cap.release()
    if overlay:
        cv2.destroyAllWindows()
    print(f"✅ Saved pose data: {output_json_path}")

def process_all_videos():
    for label in ["correct", "wrong"]:
        input_dir = os.path.join(DATA_ROOT, label)
        output_dir = os.path.join(OUTPUT_ROOT, "wrong" if "wrong" in label else "correct")

        for filename in os.listdir(input_dir):
            if filename.endswith(".mp4"):
                video_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace(".mp4", ".json"))
                print(f"🔍 Processing: {video_path}")
                extract_pose_from_video(video_path, output_path, overlay=False)  # set overlay=True for visual check

if __name__ == "__main__":
    process_all_videos()
