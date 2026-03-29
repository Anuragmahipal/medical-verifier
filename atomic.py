#!/usr/bin/env python3
import httpx
from openai import OpenAI
import sys
import os
import re
from datetime import datetime

# ---- CONFIG ----
MODEL = "phi4" # Change to "llama3.2" if you get Out of Memory errors!
TIMEOUT = 600.0
OUTPUT_DIR = "atomic_output"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(timeout=TIMEOUT)
)

def decompose_step(question, step_text, step_index):
    """
    Decompose medical text into atomic claims following paper rules.
    Generic approach works with any medical reasoning text.
    """
    system_prompt = """You are a Medical Knowledge Extractor. Your task: Break down medical reasoning text into ATOMIC CLAIMS ONLY.

AN ATOMIC CLAIM IS: A single, independent, verifiable factual sentence.

CRITICAL RULES:
1. ONE FACT PER LINE: Never use conjunctions like "and", "or", "while" to join facts. Split them!
   ✗ "Cones are in the retina and responsible for color vision."
   ✓ "Cones are located in the retina." (Line 1)
   ✓ "Cones are responsible for color vision." (Line 2)
   
2. RESOLVE PRONOUNS: Replace "It", "They", "This", or "Which" with the specific medical entity they refer to.
   ✗ "They are concentrated in the foveola."
   ✓ "Cones are concentrated in the foveola."

3. BE EXHAUSTIVE: Extract every single anatomical, physiological, and clinical fact. Do not summarize.

4. NO INFERENCE: Extract exactly what the text states. Do not add external knowledge here.

OUTPUT FORMAT:
Output exactly one atomic claim per line. Do not number the lines. Do not add introductory text."""

    user_content = f"QUESTION: {question}\n\nTEXT:\n{step_text}"

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

def extract_atomic_claims(medical_text, question=""):
    """
    Extract atomic claims from ANY medical reasoning text.
    Generic function works with diverse medical sources.
    """
    # Split text into logical chunks
    chunks = re.split(r'\n\n+|(?=\d+\.)', medical_text.strip())

    all_claims = []

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue

        atomic_segment = decompose_step(question, chunk.strip(), i)

        # Parse output from LLM
        lines = atomic_segment.split('\n')
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip section headers and meta-comments
            if any(line.upper().startswith(h) for h in [
                'INPUT:', 'OUTPUT:', 'STEP ', 'SECTION', 'NOTE:',
                'EXPLANATION:', 'ANSWER:', 'EXAMPLE'
            ]):
                continue

            # Skip uncertain/meta statements
            if any(p in line.lower() for p in [
                'to find', 'additional information', 'we determine',
                'cannot be', 'unclear', 'unknown', 'insufficient'
            ]):
                continue

            # Remove numbering (1. , 2. , - , *) if the LLM adds it anyway
            clean_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()

            if not clean_line or len(clean_line) < 5:
                continue

            all_claims.append(clean_line)

    # Remove duplicates while preserving order
    seen = set()
    unique_claims = []
    for claim in all_claims:
        claim_lower = claim.lower()
        if claim_lower not in seen:
            seen.add(claim_lower)
            unique_claims.append(claim)

    return unique_claims


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


if __name__ == "__main__":
    print("=== Medical Reasoning Analyzer ===\n")

    try:
        q = input("Enter medical question (or press Enter to skip): ").strip()

        r_full = multiline_input("Paste medical reasoning text:")
        if not r_full.strip():
            print("Error: Reasoning cannot be empty.")
            exit(1)

        # Process the text
        claims = extract_atomic_claims(r_full, q)

        print(f"\n✓ Extracted {len(claims)} atomic claims\n")
        for i, claim in enumerate(claims, 1):
            print(f"{i}. {claim}")

    except KeyboardInterrupt:
        print("\n\nCancelled.")
        exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)