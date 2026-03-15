# ontology.py

NODE_TYPES = [
    "Structure",
    "Process",
    "Condition",
    "Function",
    "Symptom",
    "Disease"
]

EDGE_TYPES = [
    "entails",
    "impairs",
    "enables",
    "occurs_in",
    "enhances_risk_of"
]

ADMISSIBILITY = {

    "occurs_in": {
        "Structure": ["Structure"],
        "Process": ["Structure","Process"],
        "Condition": ["Structure"],
        "Symptom": ["Structure"],
        "Disease": ["Structure"]
    },

    "enables": {
        "Structure": ["Structure","Function","Process","Condition"],
        "Function": ["Process","Condition"],
        "Process": ["Function","Process","Condition"],
        "Condition": ["Function","Process","Condition"],
        "Symptom": ["Function","Process","Condition"],
        "Disease": ["Function","Process","Condition"]
    },

    "impairs": {
        "Structure": ["Structure","Function","Process","Symptom","Disease"],
        "Process": ["Structure","Function","Process","Condition", "Symptom","Disease"],
        "Condition": ["Structure","Function","Process","Condition"],
        "Disease": ["Structure","Function","Process","Condition"]
    },

    "entails": {
        "Structure": ["Condition"], # REMOVED "Function" to enforce "Structures enable, they don't entail"
        "Function": ["Condition"],
        "Process": ["Process", "Condition","Symptom","Disease"], 
        "Condition": ["Process","Condition","Symptom","Disease"],
        "Disease": ["Process","Condition","Symptom","Disease"]
    },

    "enhances_risk_of": {
        "Structure": ["Condition","Disease","Symptom"],
        "Process": ["Disease"],
        "Condition": ["Symptom","Disease"],
        "Disease": ["Condition","Symptom","Disease"],
        "Symptom": ["Condition", "Disease", "Symptom"]
    }
}

def is_admissible(source_type, edge, target_type):
    """
    Check if a triplet (source_type -> edge -> target_type) is admissible.

    Returns:
        (bool, str): (is_valid, reason_message)
    """
    if edge not in ADMISSIBILITY:
        return False, f"✗ Unknown edge type: {edge}"

    if source_type not in ADMISSIBILITY[edge]:
        return False, f"✗ {source_type} cannot be source of {edge}"

    if target_type not in ADMISSIBILITY[edge][source_type]:
        return False, f"✗ {source_type} --[{edge}]--> {target_type} (type constraint violated)"

    return True, f"✓ {source_type} --[{edge}]--> {target_type}"