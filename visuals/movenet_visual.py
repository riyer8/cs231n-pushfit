import time
import cv2  # type: ignore
import os
import json
import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore

VIDEO_PATH = "datasets/videos/correct/correct1.mp4"
OUTPUT_VIDEO_PATH = "visuals/movenet_overlay.mp4"

os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# setting up movenet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

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

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = preprocess_frame(img_rgb)

        outputs = model.signatures['serving_default'](input_img)
        keypoints = outputs['output_0'].numpy()
        kps = keypoints[0][0]

        frame_keypoints = []
        for y, x, score in kps:
            abs_x = float(x * width)
            abs_y = float(y * height)
            frame_keypoints.append({
                "x": abs_x,
                "y": abs_y,
                "score": float(score)
            })
            if score > 0.3:
                cv2.circle(frame, (int(abs_x), int(abs_y)), 4, (0, 255, 0), -1)

        keypoints_all.append({"keypoints": frame_keypoints})
        cv2.imshow("MoveNet Keypoints", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Overlay video saved to: {output_video_path}")

if __name__ == "__main__":
    extract_pose_with_visual_and_save(VIDEO_PATH, OUTPUT_VIDEO_PATH)