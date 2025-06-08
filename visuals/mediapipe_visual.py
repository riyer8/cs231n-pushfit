import time
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import os
import json

VIDEO_PATH = "datasets/videos/correct/correct1.mp4"
OUTPUT_VIDEO_PATH = "visuals/mediapipe.mp4"

os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# setting up mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_pose_with_visual_and_save(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        keypoints_all.append({"keypoints": frame_keypoints})

        cv2.imshow("Pose Keypoints", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Overlay video saved to: {output_video_path}")

if __name__ == "__main__":
    extract_pose_with_visual_and_save(VIDEO_PATH, OUTPUT_VIDEO_PATH)
