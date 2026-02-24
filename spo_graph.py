import os
import json
import re
import httpx
from openai import OpenAI
from datetime import datetime

# ---------------- CONFIG ----------------
MODEL = "mistral:instruct"
INPUT_DIR = "atomic_output"
GRAPH_DIR = "reasoning_graphs"
TIMEOUT = 600.0

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(timeout=TIMEOUT)
)

def transform_to_triplets(claims_text):
    """
    Takes a list of atomic sentences and forces them into SPO triplets.
    """
    system_prompt = """You are a Knowledge Graph Engineer. 
Convert the provided medical claims into exact [Subject] -> [Predicate] -> [Object] triplets.
ALLOWED PREDICATES:
- PART_OF (for components)
- LOCATED_IN (for anatomy regions)
- SUPERFICIAL_TO / DEEP_TO (for layers)
- CONTINUOUS_WITH (for connected fascia)
- EXCLUDES (for things explicitly stated as NOT being part)
RULES:
1. Use SHORT predicates (e.g., PART_OF, LOCATED_IN, IS_A, COMPOSED_OF, NOT_PART_OF).
2. Every line must have exactly two '->' separators.
3. Keep medical terminology precise.
4. Instead of using and in any claim, break it into multiple lines.
5. Do not add information that isn't there.
6. Do not use pronouns (it, they, she). Use the actual entity names.
7. If a claim has multiple facts, break it into multiple lines.
8. If a claim is an observation (e.g., "Therefore..."), convert it to a logical triplet like EXCLUDES.


### EXAMPLE (Input from Module A):
1. The urogenital diaphragm is located in the deep perineal pouch.
2. The deep perineal pouch contains the deep transverse perineal muscle.
3. The deep perineal pouch also contains the sphincter urethrae.
4. Colle's fascia is a layer of the superficial perineal pouch.
5. The superficial perineal pouch is located superficial to the perineal membrane.
6. Therefore, Colle's fascia does not contribute to the urogenital diaphragm.

### OUTPUT TRIPLETS:
Urogenital diaphragm -> LOCATED_IN -> Deep perineal pouch
Deep perineal pouch -> CONTAINS -> Deep transverse perineal muscle
Deep perineal pouch -> CONTAINS -> Sphincter urethrae
Colle's fascia -> PART_OF -> Superficial perineal pouch
Superficial perineal pouch -> SUPERFICIAL_TO -> Perineal membrane
Colle's fascia -> EXCLUDES -> Urogenital diaphragm
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": claims_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_file_to_graph():
    # 1. Get latest file from Module A
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    if not files: return
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        content = f.read()

    # 2. Extract sections
    step_blocks = re.split(r'\[ORIGINAL STEP \d+\]:', content)
    final_graph = {"steps": []}

    for i, block in enumerate(step_blocks[1:], 1):
        # Extract the claims text
        claims_match = re.search(r'\[ATOMIC CLAIMS\]:\n(.*?)(?=\n-{3,}|\n\[|$)', block, re.DOTALL)
        if not claims_match: continue
        
        raw_claims = claims_match.group(1).strip()
        
        # 3. Call LLM to transform this specific step into triplets
        print(f"Transforming Step {i} into Triplets...")
        triplets_text = transform_to_triplets(raw_claims)
        
        parsed_triplets = []
        for line in triplets_text.split('\n'):
            parts = [p.strip() for p in line.split('->')]
            if len(parts) == 3:
                parsed_triplets.append({
                    "subject": parts[0],
                    "predicate": parts[1],
                    "object": parts[2]
                })
        
        final_graph["steps"].append({
            "step_number": i,
            "triplets": parsed_triplets
        })

    # 4. Save as JSON
    if not os.path.exists(GRAPH_DIR): os.makedirs(GRAPH_DIR)
    save_path = os.path.join(GRAPH_DIR, f"triplet_graph_{datetime.now().strftime('%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(final_graph, f, indent=2)
    
    return save_path

if __name__ == "__main__":
    path = process_file_to_graph()
    print(f"✅ Success! Structured Triplet Graph saved to: {path}")