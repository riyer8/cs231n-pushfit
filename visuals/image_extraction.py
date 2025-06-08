# gets the 1-second frame of movenet / mediapipe overlays

import cv2 # type: ignore
import os

VIDEO_PATH = "visuals/movenet.mp4"
OUTPUT_IMAGE_PATH = "visuals/movenet_image.jpg"
TARGET_TIME_SECONDS = 1.0

os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
target_frame = int(TARGET_TIME_SECONDS * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ret, frame = cap.read()
if ret:
    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
    print(f"✅ Saved frame at {TARGET_TIME_SECONDS}s to: {OUTPUT_IMAGE_PATH}")
else:
    print("❌ Failed to extract frame.")

cap.release()