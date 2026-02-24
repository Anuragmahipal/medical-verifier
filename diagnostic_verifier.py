def verify_diagnosis(graph):

    obs = set(graph["observations"])
    supports = graph["supports"]
    contradictions = graph["contradictions"]
    final = graph["final_answer"]

    support_count = 0
    contradict_count = 0

    for s in supports:
        if s["hypothesis"] == final and s["observation"] in obs:
            support_count += 1

    for c in contradictions:
        if c["hypothesis"] == final and c["observation"] in obs:
            contradict_count += 1

    if support_count == 0:
        return False, "No evidence supports diagnosis"

    if contradict_count > 0:
        return False, "Diagnosis contradicts observations"

    return True, "Best explanation supported"