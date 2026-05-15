import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
SYSTEM_INSTRUCTION = """
You are a puzzle assembly assistant. Convert spatial hints into clear vocal instructions.
Rules:
1. Exactly one instruction per piece.
2. NEVER use the word "next". Use "Then", "Now", "Following that".
3. Output ONLY a raw JSON array of strings.
"""
vlm_model = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    system_instruction=SYSTEM_INSTRUCTION
)

fragment_data = [
    {"piece_index": 1, "direction_hint": "Move right and down to row 3, column 4"},
    {"piece_index": 2, "direction_hint": "Move left to row 1, column 1"}
]

hints = [f"Piece {p['piece_index']}: {p['direction_hint']}" for p in fragment_data]
prompt = f"Spatial Data:\n{hints}"

import time
try:
    start_time = time.time()
    response = vlm_model.generate_content(prompt)
    end_time = time.time()
    text = response.text.strip()
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("Gemini Raw Response:")
    print(text)
    
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.replace("```", "", 1).rsplit("```", 1)[0].strip()
        
    steps = json.loads(text)
    print("Parsed JSON steps:")
    print(steps)
except Exception as e:
    print(f"Error: {e}")
