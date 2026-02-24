import httpx
from openai import OpenAI
import sys
import os
import re
from datetime import datetime

# ---------------- CONFIG ----------------
MODEL = "mistral:instruct"
TIMEOUT = 600.0
OUTPUT_DIR = "atomic_output"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(timeout=TIMEOUT)
)

def decompose_step(question, step_text, step_index):
    """
    Processes a single segment of reasoning to maintain high focus and granularity.
    """
    system_prompt = """You are a Medical Logic Auditor. 
Your task is to break down a specific medical reasoning step into a numbered list of ATOMIC CLAIMS.

RULES:
1. One clinical fact per line.
2. Do not use pronouns (it, they, she). Use the actual entity names.
3. Keep the original sequence of the reasoning.
4. Strict Literalism: Only extract facts that are EXPLICITLY STATED in the text. Do not add medical knowledge. If the text says 'it is a structure,' do not name the structure yourself.
5. If a line has multiple facts, break it into multiple lines.
6. Do not include conversational filler (e.g., "The answer is", "Therefore").
7. Try to break it down as much as possible, even if it seems obvious.

EXAMPLES:
Input: "The heart has four chambers and is located in the mediastinum."
Output:
1. The heart has four chambers.
2. The heart is located in the mediastinum."
"""

    user_content = f"QUESTION: {question}\n\nREASONING STEP {step_index}: {step_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in Step {step_index}: {str(e)}"

def save_output(full_content):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(OUTPUT_DIR, f"sequential_{timestamp}.txt")

    with open(filepath, "w") as f:
        f.write(full_content)
    return filepath

def multiline_input(prompt):
    print(prompt)
    print("(Paste text, then press Ctrl+D [Linux] or Ctrl+Z [Windows] and Enter)\n")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        return "\n".join(lines)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("--- Module A: Sequential Medical Fact Decomposer ---")
    
    q = multiline_input("Enter the Medical Question: ")
    r_full = multiline_input("Paste the Reasoning Steps (Numbered 1, 2, 3...): ")

    # Regex to split by numbers (1. , 2. ) or the word "Conclusion:"
    # This ensures each paragraph/step is a separate item in the list
    steps = re.split(r'\n(?=\d+\.|\bConclusion:)', r_full.strip())
    
    final_output_parts = [f"QUESTION: {q}\n", "="*40]
    
    print(f"\nDetected {len(steps)} segments. Starting sequential processing...")

    for i, step in enumerate(steps, 1):
        if not step.strip(): continue
        
        print(f"Processing Segment {i}...")
        atomic_segment = decompose_step(q, step.strip(), i)
        
        # Formatting for the final text file
        final_output_parts.append(f"\n[ORIGINAL STEP {i}]:\n{step.strip()}")
        final_output_parts.append(f"\n[ATOMIC CLAIMS]:\n{atomic_segment}")
        final_output_parts.append("-" * 30)

    # Combine everything and save
    full_report = "\n".join(final_output_parts)
    saved_path = save_output(full_report)
    
    print(f"\n✅ Success! Sequential atomic claims saved to: {saved_path}")
    print("\n--- PREVIEW ---")
    print(full_report[:600] + "...")