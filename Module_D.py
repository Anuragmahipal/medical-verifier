import json
import networkx as nx
import matplotlib.pyplot as plt
import os


class GraphVisualizer:
    def __init__(self, json_data):
        self.data = json_data
        self.G = nx.MultiDiGraph()
        self.build_graph()

    # ---------------- GRAPH BUILDER ----------------
    def clean_entity(self, text):
        """Normalize entity names for consistent graph nodes"""
        text = text.split('. ')[-1]
        text = text.split(' (')[0]
        return text.lower().strip()

    def build_graph(self):
        print("\n--- Building Knowledge Graph ---")

        for step in self.data['steps']:
            step_no = step.get("step_number", "?")

            for triplet in step['triplets']:
                sub = self.clean_entity(triplet['subject'])
                obj = self.clean_entity(triplet['object'])
                pred = triplet['predicate'].upper().strip()

                # Skip vague placeholders
                if "unspecified" in sub or "unspecified" in obj:
                    continue

                # Convert negative statements into logical edges
                if "not" in obj:
                    obj = obj.replace("not ", "")
                    pred = "EXCLUDES"

                self.G.add_edge(sub, obj, label=pred, step=step_no)
                print(f"Step {step_no}: {sub} --[{pred}]--> {obj}")

        print(f"\nGraph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges\n")

    # ---------------- VISUALIZATION ----------------
    def visualize(self, output_path="reasoning_graph.png"):
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(self.G, k=0.7, seed=42)

        nx.draw_networkx_nodes(self.G, pos, node_color="lightblue", node_size=2600)
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        # draw edges
        nx.draw_networkx_edges(self.G, pos, arrows=True)

        # edge labels
        edge_labels = {}
        for u, v, key, data in self.G.edges(keys=True, data=True):
            edge_labels[(u, v)] = data['label']

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7)

        plt.title("Medical Reasoning Knowledge Graph", fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ Graph saved to {output_path}")
        plt.show()

    # ---------------- REASONING CHECK ----------------
    def check_exclusion_path(self, start_node, target_node):
        start_node = start_node.lower()
        target_node = target_node.lower()

        print(f"\n--- Checking logical path: {start_node} → {target_node} ---")

        try:
            paths = list(nx.all_simple_paths(self.G, start_node, target_node))

            if not paths:
                print("No reasoning path found")
                return False

            for path in paths:
                print("Path:", " → ".join(path))

                for i in range(len(path) - 1):
                    edges = self.G.get_edge_data(path[i], path[i+1])

                    for edge_id in edges:
                        label = edges[edge_id]['label']

                        if label in ["EXCLUDES", "NOT_PART_OF"]:
                            print(f"❌ Logical contradiction via {label}")
                            return False

            print("✅ Reasoning path valid")
            return True

        except nx.NetworkXNoPath:
            print("No path exists in reasoning graph")
            return False


# ---------------- RUNNER ----------------
if __name__ == "__main__":

    # load Module-B output JSON
    with open("reasoning_graphs/triplet_graph_122316.json", "r") as f:
        data = json.load(f)

    visualizer = GraphVisualizer(data)

    # create graph image
    visualizer.visualize()

    # Example: verify reasoning claim
    start = input("\nEnter starting entity: ")
    target = input("Enter target entity: ")

    result = visualizer.check_exclusion_path(start, target)

    print("\n====== FINAL VERDICT ======")
    if result:
        print("✔ Reasoning is LOGICALLY CONSISTENT")
    else:
        print("✘ Reasoning is CONTRADICTORY or INCOMPLETE")