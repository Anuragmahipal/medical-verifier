import json
import os
from datetime import datetime

# ---------------- CONFIG ----------------
GRAPH_INPUT_DIR = "reasoning_graphs"
JUDGE_OUTPUT_DIR = "logic_reports"

class LogicJudge:
    def __init__(self, graph_data):
        self.graph = graph_data
        self.contradictions = []

    def is_placeholder(self, text):
        """Identifies if a node is just a placeholder for unknown info."""
        placeholders = ["unspecified", "unknown", "multiple structures", "not mentioned"]
        return any(p in text.lower() for p in placeholders)

    def check_internal_consistency(self):
        definitions = {} # {subject: {step_number: set(objects)}}

        for step in self.graph['steps']:
            step_num = step['step_number']
            for triplet in step['triplets']:
                sub = triplet['subject'].split('. ')[-1].lower().strip()
                pred = triplet['predicate']
                obj = triplet['object'].lower().strip()

                # We only track structural definitions
                if pred in ["COMPOSED_OF", "PART_OF", "INCLUDES", "IS_A"]:
                    if sub not in definitions:
                        definitions[sub] = {}
                    if step_num not in definitions[sub]:
                        definitions[sub][step_num] = set()
                    
                    # Only add if it's NOT a placeholder
                    if not self.is_placeholder(obj):
                        definitions[sub][step_num].add(obj)

        for sub, steps in definitions.items():
            step_keys = sorted(steps.keys())
            for i in range(len(step_keys) - 1):
                prev_step = step_keys[i]
                curr_step = step_keys[i+1]
                
                prev_vals = steps[prev_step]
                curr_vals = steps[curr_step]

                # LOGIC: If both steps have real data, they must overlap.
                # If one is empty (because it was a placeholder), we allow it (Information Gain).
                if prev_vals and curr_vals:
                    if not (prev_vals & curr_vals):
                        self.contradictions.append({
                            "entity": sub,
                            "conflict": f"Step {prev_step} defined as {prev_vals}, but Step {curr_step} changed to {curr_vals}."
                        })

    def run_judge(self):
        self.check_internal_consistency()
        status = "PASSED" if not self.contradictions else "FAILED"
        return {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "contradictions_found": self.contradictions,
            "summary": f"Logic check {status}. Found {len(self.contradictions)} actual conflicts."
        }

# ... [Standard Main block to load JSON and save report] ...