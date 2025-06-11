import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv
import os
from feature_extraction import analyze_video_pose

# Constants
DEFAULT_MODEL = "models/gemini-1.5-flash"
PROMPT_FILE = "llm_pipeline/prompt_engineering.txt"
OUTPUT_FILE = "llm_pipeline/output.txt"

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found (set env var or .env file).")

# Configure Gemini
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

def gemini_integration(feedback_file):
    print("📄 Loading input files...")
    prompt_text = load_text(PROMPT_FILE)
    feedback_text = load_text(feedback_file)

    combined_prompt = f"{prompt_text.strip()}\n\n{feedback_text.strip()}"

    print("🤖 Generating response from Gemini...")
    response_text = generate_feedback(combined_prompt)

    print("💾 Saving output to output.txt")
    save_text(OUTPUT_FILE, response_text)

    print("✅ Done!")

if __name__ == "__main__":
    JSON_PATH = "llm_pipeline/keypoints_wrong3.json"
    OUTPUT_TXT = "llm_pipeline/feedback_wrong3.txt"
    TXT_FILE = analyze_video_pose(JSON_PATH, OUTPUT_TXT)
