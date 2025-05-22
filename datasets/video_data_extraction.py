import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pose_estimation import pose_estimation

video_base_path = "datasets/exercise_video_data"
output_json_path = "datasets/exercise_json_data"

os.makedirs(output_json_path, exist_ok=True)

exercise_data = {
    "bench_press": 61,
    "lat_pulldown": 51,
    "push_up": 56,
    "tricep_pushdown": 50
}

for exercise, count in exercise_data.items():
    for i in range(1, count + 1):
        file_name = f"{exercise}_{i}.mp4"
        folder_name = exercise
        video_path = os.path.join(video_base_path, folder_name, file_name)
        output_dir = os.path.join(output_json_path, exercise)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file_name.replace(".mp4", ".json"))

        # Check if video exists
        if os.path.exists(video_path):
            print(f"🚀 Processing {video_path}")
            pose_estimation(video_path, file_name, output_path)
        else:
            print(f"⚠️ Video not found: {video_path}")
