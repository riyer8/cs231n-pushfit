# movenet extractor for a single video

import os
import json
import cv2  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore

VIDEO_PATH = "llm_pipeline/wrong3.mp4"

# Load the MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

def video_to_json_path(video_path: str) -> str:
    dir_path, filename = os.path.split(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    keypoints_filename = f"keypoints_{name_without_ext}.json"
    return os.path.join(dir_path, keypoints_filename)

def extract_keypoints_to_json(video_path):
    output_json_path = video_to_json_path(video_path)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        keypoints_all.append({"keypoints": frame_keypoints})

    cap.release()

    with open(output_json_path, 'w') as f:
        json.dump(keypoints_all, f, indent=2)

    print(f"✅ Keypoints saved to: {output_json_path}")
    return output_json_path

# test example
# extract_keypoints_to_json(VIDEO_PATH, KEYPOINTS_OUTPUT_PATH)