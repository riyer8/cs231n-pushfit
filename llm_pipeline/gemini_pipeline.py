import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv
import os
from feature_extraction import analyze_video_pose
from classification import predict_single_video
from math import atan2, degrees
from movenet import extract_keypoints_to_json
from feature_extraction import keypoints_to_feedback_path

DEFAULT_MODEL = "models/gemini-1.5-flash"
PROMPT_FILE = "llm_pipeline/prompt_engineering.txt"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found (set env var or .env file).")

genai.configure(api_key=api_key)

def load_text(path):
    with open(path, "r") as f:
        return f.read()

def save_text(path, content):
    with open(path, "w") as f:
        f.write(content)

def generate_feedback(prompt_text):
    model = genai.GenerativeModel(DEFAULT_MODEL)
    response = model.generate_content(prompt_text)
    return response.text.strip()

def gemini_integration(feedback_file, output_file):
    print("📄 Loading input files...")
    prompt_text = load_text(PROMPT_FILE)
    feedback_text = load_text(feedback_file)

    combined_prompt = f"{prompt_text.strip()}\n\n{feedback_text.strip()}"

    print("🤖 Generating response from Gemini...")
    response_text = generate_feedback(combined_prompt)

    print("💾 Saving output to output.txt")
    save_text(output_file, response_text)

    print("✅ Done!")

def end_to_end_integration(video_path):
    json_path = extract_keypoints_to_json(video_path)
    feedback_txt = keypoints_to_feedback_path(json_path)
    txt_file = analyze_video_pose(json_path, feedback_txt)
    output_file = "llm_pipeline/output.txt"
    gemini_integration(txt_file, output_file)

VIDEO_PATH = "llm_pipeline/wrong3.mp4"
end_to_end_integration(VIDEO_PATH) # produced in llm_pipeline/output.txt