#!/usr/bin/env python3
import os
import json
import re
import httpx
from openai import OpenAI
from datetime import datetime
from ontology import is_admissible

# --- IMPORTS FOR SCISPACY (UMLS SETUP) ---
import spacy
from scispacy.linking import EntityLinker
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- THE ADVANCED SYNTACTIC CLASSIFIER CLASS ---
class MedicalOntologyClassifier:

    """
    Classifies medical entities into ontology types:
    {Structure, Process, Condition, Function, Symptom, Disease}.

    The classification follows a hybrid pipeline:

    1. NLP-based heuristics (verbs, action nouns → Process)
    2. Pharmacology heuristics (drug detection → Structure)
    3. UMLS lookup via SciSpacy (high-confidence mapping)
    4. LLM fallback ("oracle") for unknown terms

    This layered approach ensures:
    - Speed (heuristics first)
    - Accuracy (UMLS grounding)
    - Coverage (LLM fallback)

    Attributes:
        nlp: SciSpacy pipeline with UMLS linker
        linker: UMLS entity linker
        client: LLM client (Ollama / OpenAI-compatible)
        model: LLM model name
        cache: Stores previously classified entities for efficiency
    """

    def __init__(self, llm_client, model_name):
        """
        Initializes the classifier by loading:
        - SciSpacy model (en_core_sci_sm)
        - UMLS entity linker

        Args:
            llm_client: Client used to query LLM (acts as fallback classifier)
            model_name: Name of the LLM model (e.g., "phi4", "llama3")

        Note:
            First-time initialization may take time due to UMLS data loading (~500MB).
        """


        print("Loading Medical Ontology Database (UMLS) and Dependency Parser...")
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")
        
        # Pass the OpenAI client so the classifier can use the LLM Oracle
        self.client = llm_client
        self.model = model_name
        print("Database and Parser loaded successfully.")

        self.tui_mapping = {
            "T121": "Structure", "T109": "Structure", "T116": "Structure", 
            "T122": "Structure", "T123": "Structure", "T200": "Structure", 
            "T017": "Structure", "T022": "Structure", "T023": "Structure", 
            "T024": "Structure", "T025": "Structure", "T074": "Structure",
            "T047": "Disease", "T048": "Disease", "T191": "Disease", "T050": "Disease",
            "T184": "Symptom", "T033": "Symptom", "T034": "Symptom",
            "T059": "Process", "T060": "Process", "T061": "Process", 
            "T038": "Process", "T044": "Process",
            "T039": "Function", "T040": "Function", "T041": "Function",
            "T046": "Condition", "T032": "Condition", "T037": "Condition"
        }
        # This cache will now automatically build itself!
        self.cache = {}

    def _ask_llm_oracle(self, word):
        """Asks the LLM to classify a single unknown word and strict-formats the output."""
        prompt = f"""Classify the medical term '{word}' into EXACTLY ONE of these types: 
        Structure, Process, Condition, Function, Symptom, Disease. 
        If it represents an action, change, or treatment, output Process.
        Output ONLY the exact category name. No other text."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ans = response.choices[0].message.content.strip().capitalize()
            if ans in {'Structure', 'Process', 'Condition', 'Function', 'Symptom', 'Disease'}:
                return ans
            return "Condition" # Failsafe
        except:
            return "Condition"

    def classify(self, entity_text, fallback_type="Condition"):
        """
        Classifies a medical entity into a predefined ontology type.

        Pipeline:
            1. Cache lookup (fastest path)
            2. Dependency parsing (identify root token)
            3. NLP heuristics (verbs/action nouns → Process)
            4. Drug detection heuristics → Structure
            5. UMLS lookup (via SciSpacy linker)
            6. LLM fallback (oracle)

        Args:
            entity_text (str): Input medical entity
            fallback_type (str): Default type if classification fails

        Returns:
            str: Classified ontology type

        Notes:
            - Uses root token for semantic classification
            - Caches results to avoid recomputation
        """
        entity_text = str(entity_text).strip()
        if not entity_text:
            return fallback_type
            
        text_lower = entity_text.lower().strip()
        if text_lower in self.cache:
            return self.cache[text_lower]

        doc = self.nlp(entity_text)
        root_token = next((token for token in doc if token.head == token), None)

        if not root_token:
            return fallback_type

        # ==========================================
        # 1. PURE NLP & ACTION NOUN CATCHER
        # ==========================================
        action_nouns = ['counteraction', 'reduction', 'induction', 'treatment', 'management', 'assessment', 'use', 'option']
        if root_token.pos_ == "VERB" or any(w in text_lower for w in action_nouns):
            self.cache[text_lower] = "Process"
            return "Process"

        # ==========================================
        # 2. PHARMACOLOGY SAFETY NET (Crucial for drugs!)
        # ==========================================
        drug_keywords = ['analogue', 'pill', 'inhibitor', 'drug', 'medication', 'gonadorelin']
        drug_suffixes = ['ide', 'lin', 'ole', 'gen', 'one', 'ins']
        
        last_word = text_lower.split()[-1] if text_lower else ""
        if any(w in text_lower for w in drug_keywords) or any(last_word.endswith(s) for s in drug_suffixes):
            self.cache[text_lower] = "Structure"
            return "Structure"

        # ==========================================
        # 3. UMLS DATABASE LOOKUP
        # ==========================================
        target_ent = None
        for ent in doc.ents:
            if ent.start <= root_token.i < ent.end:
                target_ent = ent
                break

        if target_ent and target_ent._.kb_ents:
            best_match_cui = target_ent._.kb_ents[0][0]
            concept = self.linker.kb.cui_to_entity[best_match_cui]
            for tui in concept.types:
                if tui in self.tui_mapping:
                    result = self.tui_mapping[tui]
                    self.cache[text_lower] = result
                    return result

        # ==========================================
        # 4. THE LLM ORACLE
        # ==========================================
        oracle_decision = self._ask_llm_oracle(root_token.text)
        self.cache[text_lower] = oracle_decision
        return oracle_decision

# Note: You now initialize the classifier AFTER setting up the 'client' in your main code
# ---- CONFIG ----
MODEL = "phi4"
INPUT_DIR = "atomic_output"
GRAPH_DIR = "reasoning_graphs"
TIMEOUT = 600.0

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(timeout=TIMEOUT)
)

# Initialize the classifier AFTER the client is defined, and pass the required arguments
classifier = MedicalOntologyClassifier(client, MODEL)

def transform_to_triplets(claims_text):
    system_prompt = """You are an expert Medical Knowledge Graph Engineer. 
Convert the provided list of medical atomic claims into a strict JSON object representing a causal reasoning graph.

# ONTOLOGY: NODE TYPES
- Structure: Physical entities (e.g., organs, cells, drugs, pills).
- Function: Physiological capabilities (e.g., color vision, immunity).
- Process: Active sequences; clinical treatments, surgeries, estrogen production, biological growth.
- Condition: A state of being, clinical status, or abnormality (e.g., tissue growth, hypoestrogenic state).
- Symptom: Subjective/observable evidence of a physical disturbance (e.g., pain, infertility).
- Disease: A defined pathological category (e.g., endometriosis).

# ONTOLOGY: RELATIONS (Pick EXACTLY ONE)
- entails: Necessary consequence or causation. (Process/Condition/Disease causes another state).
- impairs: Disruption, damage, or reduction. (Treatments impair Diseases/Symptoms/Conditions).
- enables: Facilitation/Capability. (Structures ENABLE processes/functions).
- occurs_in: Spatial localization or composition. 
- enhances_risk_of: Probability increase.

# CRITICAL RULES FOR EXTRACTION:
1. CANONICAL MEDICAL KEYWORDS (1-3 WORDS): Abstract long, conversational phrases into formal, scientific medical terms. DO NOT copy-paste long chunks of text. 
   - ✗ Bad: "Growth of tissue outside the uterus" -> ✓ Good: "Ectopic tissue growth" (Condition)
   - ✗ Bad: "Tissue similar to the lining inside the uterus" -> ✓ Good: "Endometrial tissue" (Structure)
   - ✗ Bad: "Medical treatments for endometriosis" -> ✓ Good: "Medical treatment" (Process)
   - ✗ Bad: "Symptoms of endometriosis" -> ✓ Good: "Pain" or "Symptom"
2. ATOMIZE LISTS & PARENTHESES: NEVER group entities. Strip out parentheses.
   - ✗ Bad: "Medications (GnRH analogues)" -> ✓ Good: "GnRH analogue"
   - ✗ Bad: "Aromatase inhibitors (Letrozole)" -> ✓ Good: "Letrozole"
3. THE TRANSLATION KEY (STRICT): You CANNOT invent verbs. Translate them:
   - "causes", "leads to" -> use `entails`
   - "reduces", "downregulates", "alleviates", "manages" -> use `impairs`
   - "used in", "allows" -> use `enables`
4. CAUSATION LOGIC: 
   - Tissue growth *causes* pain/infertility, so it `entails` them.
   - Drugs (Structures) `enable` treatments. They DO NOT `entail` them.

# OUTPUT FORMAT
You MUST output valid JSON matching this exact schema:
{
  "triplets": [
    {
      "subject": "string",
      "subject_type": "Structure|Function|Process|Condition|Symptom|Disease",
      "predicate": "entails|impairs|enables|occurs_in|enhances_risk_of",
      "object": "string",
      "object_type": "Structure|Function|Process|Condition|Symptom|Disease"
    }
  ]
}
Output ONLY JSON."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"INPUT CLAIMS:\n{claims_text}"}
            ],
            response_format={"type": "json_object"}, 
            temperature=0
        )
        
        raw_output = response.choices[0].message.content.strip()
        raw_output = re.sub(r'^```json\s*', '', raw_output)
        raw_output = re.sub(r'\s*```$', '', raw_output)
        
        return json.loads(raw_output)
        
    except Exception as e:
        print(f"Error during LLM JSON generation: {str(e)}")
        return {"triplets": []}
    
def parse_typed_triplet(line):
    line = line.strip()
    if not line or '->' not in line:
        return None

    try:
        line = re.sub(r'^[-*\s]+', '', line).strip()
        parts = line.split('->')
        if len(parts) != 3:
            return None

        # Clean subject and object to remove LLM-generated brackets []
        subject_part = parts[0].strip()
        edge_part = parts[1].strip()
        object_part = parts[2].strip()

        subject_match = re.search(r'^(.*?)\s*\(\s*(.*?)\s*\)$', subject_part.strip())
        if not subject_match: return None
        subject = subject_match.group(1).replace('[', '').replace(']', '').strip()
        subject_type = subject_match.group(2).strip()

        edge_type = edge_part.lower().strip().replace(' ', '_')
        valid_edges = {'entails', 'impairs', 'enables', 'occurs_in', 'enhances_risk_of'}
        if edge_type not in valid_edges: return None

        object_match = re.search(r'^(.*?)\s*\(\s*(.*?)\s*\)$', object_part.strip())
        if not object_match: return None
        obj = object_match.group(1).replace('[', '').replace(']', '').strip()
        object_type = object_match.group(2).strip()

        return {
            "subject": subject, "subject_type": subject_type,
            "predicate": edge_type,
            "object": obj, "object_type": object_type
        }
    except Exception:
        return None

def process_reasoning_to_graph(reasoning_text, question=""):
    from atomic import extract_atomic_claims

    print("Step 1: Extracting atomic claims...")
    atomic_claims = extract_atomic_claims(reasoning_text, question)
    
    claims_formatted = "\n".join([f"- {claim}" for claim in atomic_claims])
    
    print("Step 2: Transforming to JSON triplets...")
    extracted_data = transform_to_triplets(claims_formatted)
    raw_triplets = extracted_data.get("triplets", [])
    
    print(f"  ✓ Generated {len(raw_triplets)} candidate triplets")

    print("Step 3: Validating and routing through UMLS/NLP...")
    parsed_triplets = []
    verification_summary = {"total": 0, "valid": 0, "invalid": 0}

    for triplet_dict in raw_triplets:
        # Failsafe: Ensure the LLM didn't miss any keys in the JSON schema
        if not all(k in triplet_dict for k in ['subject', 'subject_type', 'predicate', 'object', 'object_type']):
            continue
            
        # Clean up relation in case the LLM added spaces or caps
        triplet_dict['predicate'] = triplet_dict['predicate'].lower().strip().replace(' ', '_')

        # Advanced Root-Based Classification Override (Trust but Verify)
        triplet_dict['subject_type'] = classifier.classify(
            triplet_dict['subject'], fallback_type=triplet_dict['subject_type'] 
        )
        triplet_dict['object_type'] = classifier.classify(
            triplet_dict['object'], fallback_type=triplet_dict['object_type']
        )

        is_valid, reason = is_admissible(
            triplet_dict['subject_type'], triplet_dict['predicate'], triplet_dict['object_type']
        )

        triplet_dict['admissible'] = is_valid
        triplet_dict['admissibility_reason'] = reason

        source_claim = ""
        for claim in atomic_claims:
            if triplet_dict['subject'].lower() in claim.lower() and triplet_dict['object'].lower() in claim.lower():
                source_claim = claim
                break

        triplet_dict['source_claim'] = source_claim
        parsed_triplets.append(triplet_dict)

        verification_summary['total'] += 1
        if is_valid: verification_summary['valid'] += 1
        else: verification_summary['invalid'] += 1

    graph = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_length": len(reasoning_text),
            "parsed_claims_count": len(atomic_claims),
            "question": question
        },
        "atomic_claims": atomic_claims,
        "triplets": parsed_triplets,
        "summary": verification_summary
    }

    return graph

def multiline_input_from_user(prompt):
    print(prompt)
    print("(Paste text, then press Ctrl+D [Linux] or Ctrl+Z [Windows] and Enter)\n")
    lines = []
    try:
        while True: lines.append(input())
    except EOFError:
        return "\n".join(lines)

if __name__ == "__main__":
    print("=" * 70)
    print("MEDICAL REASONING VERIFIER")
    print("=" * 70)

    try:
        question = input("\n[Optional] Enter medical question: ").strip()
        reasoning_text = multiline_input_from_user("\nPaste medical reasoning:")

        if not reasoning_text.strip():
            print("\n❌ Error: Reasoning cannot be empty.")
            exit(1)

        print("\n" + "=" * 70)
        print("PROCESSING...")
        print("=" * 70)

        graph = process_reasoning_to_graph(reasoning_text, question)

        if not os.path.exists(GRAPH_DIR): os.makedirs(GRAPH_DIR)
        save_path = os.path.join(GRAPH_DIR, f"typed_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        with open(save_path, 'w') as f:
            json.dump(graph, f, indent=2)

        print(f"\n✅ VERIFICATION COMPLETE. Output saved to: {save_path}\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")