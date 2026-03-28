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
    system_prompt = """You are a Knowledge Graph Engineer specializing in clinical ontology.
Convert medical claims into exact [Subject] -> [Predicate] -> [Object] triplets based on a strict fixed ontology.

1. STRICT NODE TYPES (Domains)
Every Subject and Object must belong to one of these categories. If an entity does not fit one of these, drop the claim:

    Structure: Physical entities (organs, tissues, anatomical regions).

    Process: Activities or changes unfolding over time.

    Condition: A state of the system that may hold or fail.

    Function: A capability of a system.

    Symptom: An observable manifestation.

    Disease: A diagnostic abstraction.

2. STRICT EDGE TYPES (Predicates)
Use only these exact labels. Do not use synonyms or variations:

    Entails: Existence of A brings about existence of B.

    impairs: A reduces or prevents B.

    enables: A is required for B.

    occurs in: A is spatially localized to B.

    enhances risk of: The presence of A increases the probability of B.

    

Enforce Silent Filtering: Tell the model: "If a claim does not meet the node/edge criteria, SILENTLY DROP it. Do not mention the drop, do not provide notes, and do not use numbering."

Define Structural Mapping: Anatomical text often uses "is part of" or "composed of." Since your ontology doesn't allow these, explicitly instruct the model: "Map 'is part of', 'composed of', and 'contains' to the predicate 'occurs in'."

Strict Format Rule: "The output must contain ONLY the triplets in the format Subject -> Predicate -> Object. Any other text is a failure."


RULES:

    Strict Filtering: If a claim involves a concept that is not one of the 6 Node Types, do not generate a triplet for it.

    Formatting: Every line must have exactly two -> separators.

    Atomicity: If a claim has multiple facts or uses "and," break it into multiple lines.

    No Pronouns: Use actual entity names for every node.

    No Extrapolation: Do not add information not explicitly stated in the source text.

    No Synonyms or Variations: Use only the specified predicates. Do not use "causes," "is associated with," or any other wording. 


NEGATIVE CONSTRAINTS (DO NOT DO THE FOLLOWING)
- DO NOT use "is a", "part of", "composed of", or "belongs to".
- DO NOT create nodes for meta-categories (e.g., do not write "X -> is a -> Structure").
- DO NOT use "does not" or "is not". If a relationship is exclusionary and doesn't fit "impairs," DROP it.
- DO NOT include "implied" subjects. If the text is vague, drop the triplet.
- DO NOT use pronouns. Use the full entity name.


EXAMPLE:
Input: "Chronic hypertension is a condition that leads to Heart Failure. Heart Failure is a disease that occurs in the heart muscle and reduces the pumping capability of the heart."

Output Triplets:
Chronic hypertension -> enhances risk of -> Heart Failure
Heart Failure -> occurs in -> Heart muscle
Heart Failure -> impairs -> Pumping capability
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