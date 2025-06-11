import time
import cv2 # type: ignore
import os
import json
import numpy as np
import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore

DATA_ROOT = "datasets/videos"
OUTPUT_ROOT = "datasets/json/movenet/"

os.makedirs(os.path.join(OUTPUT_ROOT, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "wrong"), exist_ok=True)

# Load MoveNet Thunder model from TF Hub (more accurate)
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

def extract_pose_from_video(video_path, output_json_path, display=False):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = preprocess_frame(img_rgb)

        # Run model inference
        outputs = model.signatures['serving_default'](input_img)
        keypoints = outputs['output_0'].numpy()  # shape (1,1,17,3)

        # Parse keypoints for 17 body parts: [y, x, score]
        kps = keypoints[0][0]  # (17,3)

        frame_keypoints = []
        h, w, _ = frame.shape
        for kp in kps:
            y, x, score = kp
            frame_keypoints.append({
                "x": float(x * w),
                "y": float(y * h),
                "score": float(score)
            })

        keypoints_all.append({"keypoints": frame_keypoints})

        if display:
            # Optional: Draw keypoints on frame
            for kp in frame_keypoints:
                if kp['score'] > 0.3:
                    cv2.circle(frame, (int(kp['x']), int(kp['y'])), 4, (0,255,0), -1)
            cv2.imshow("MoveNet Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    with open(output_json_path, "w") as f:
        json.dump(keypoints_all, f, indent=2)

    cap.release()
    if display:
        cv2.destroyAllWindows()
    print(f"✅ Saved MoveNet data: {output_json_path}")

def process_all_videos():
    start_time = time.time()  # Start timer here

    for label in ["correct", "wrong"]:
        input_dir = os.path.join(DATA_ROOT, label)
        output_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith(".mp4"):
                video_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace(".mp4", ".json"))
                print(f"🔍 Processing: {video_path}")
                extract_pose_from_video(video_path, output_path, display=False)

    elapsed_time = time.time() - start_time
    print(f"⏱️ MoveNet Processing Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    process_all_videos()