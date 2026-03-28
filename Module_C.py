import requests
import json
import os
import time

API_KEY = "YOUR_UMLS_API_KEY"
BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
VERSION = "current"
CACHE_FILE = "umls_cache.json"


class ModuleCValidator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = self._load_cache()

    # ---------------- CACHE ----------------
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        return {"cuis": {}, "relations": {}}

    def save_cache(self):
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f, indent=2)

    # ---------------- UTIL ----------------
    def _sleep(self):
        time.sleep(0.2)   # prevent rate limiting

    # ---------------- CUI SEARCH ----------------
    def get_cui(self, term):
        term = term.lower().strip()

        if term in self.cache["cuis"]:
            return self.cache["cuis"][term]

        url = f"{BASE_URL}/search/{VERSION}"
        params = {"string": term, "apiKey": self.api_key}

        r = requests.get(url, params=params)
        self._sleep()

        if r.status_code == 200:
            results = r.json().get("result", {}).get("results", [])
            if results:
                cui = results[0]["ui"]
                self.cache["cuis"][term] = cui
                return cui

        self.cache["cuis"][term] = None
        return None

    # ---------------- RELATION CHECK ----------------
    def verify_relation(self, s_cui, o_cui):

        if not s_cui or not o_cui:
            return False, "CUI_MISSING"

        key = f"{s_cui}->{o_cui}"
        if key in self.cache["relations"]:
            return True, self.cache["relations"][key]

        url = f"{BASE_URL}/content/{VERSION}/CUI/{s_cui}/relations"
        params = {"apiKey": self.api_key}

        r = requests.get(url, params=params)
        self._sleep()

        if r.status_code == 200:
            for rel in r.json().get("result", []):

                related_url = rel.get("relatedId", "")
                related_cui = related_url.split("/")[-1]

                if related_cui == o_cui:
                    label = rel.get("relationLabel", "RELATED")
                    self.cache["relations"][key] = label
                    return True, label

        self.cache["relations"][key] = "NO_RELATION"
        return False, "NO_RELATION"


# ---------------- RUNNER ----------------
def validate_graph(json_path):

    validator = ModuleCValidator(API_KEY)

    with open(json_path, "r") as f:
        graph = json.load(f)

    print("\n--- UMLS VALIDATION START ---\n")

    for step in graph["steps"]:
        for triplet in step["triplets"]:

            s = triplet["subject"]
            o = triplet["object"]

            s_cui = validator.get_cui(s)
            o_cui = validator.get_cui(o)

            valid, label = validator.verify_relation(s_cui, o_cui)

            triplet["verification"] = {
                "umls_valid": valid,
                "relation_label": label,
                "subject_cui": s_cui,
                "object_cui": o_cui
            }

            print(f"{s} -> {o} : {label}")

    validator.save_cache()

    output = "verified_" + os.path.basename(json_path)
    with open(output, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"\n✅ Verified graph saved to {output}")


# -------------- MAIN --------------
if __name__ == "__main__":
    validate_graph("reasoning_graphs/triplet_graph_122316.json")