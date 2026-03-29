#!/usr/bin/env python3
"""
Graph Approach - Enhanced Visualization
Visualizes reasoning graphs with improved networking layout, edge labels, and node colors.
Works with output from spo_graph.py and saves time-distinguished image files.
"""

import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import colorsys
from datetime import datetime

GRAPH_DIR = "reasoning_graphs"

# Node type colors (distinct HSL colors)
NODE_COLORS = {
    "Structure": "#FF6B6B",      # Red
    "Process": "#4ECDC4",        # Teal
    "Condition": "#FFE66D",      # Yellow
    "Function": "#95E1D3",       # Mint
    "Symptom": "#F38181",        # Pink
    "Disease": "#AA96DA",        # Purple
}

# Edge type styles
EDGE_STYLES = {
    "entails": {"style": "solid", "width": 2.0},
    "impairs": {"style": "dashed", "width": 2.0},
    "enables": {"style": "solid", "width": 1.5},
    "occurs_in": {"style": "dotted", "width": 1.5},
    "enhances_risk_of": {"style": "dashed", "width": 2.0},
}


def visualize_graph(json_path, output_path):
    """
    Load reasoning graph JSON and create enhanced visualization
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found - {json_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON - {json_path}")
        return False

    # Create directed graph
    G = nx.DiGraph()

    triplets = data.get("triplets", [])
    if not triplets:
        print("⚠️  Warning: No triplets found in graph")
        return False

    # Add nodes and edges with attributes
    for triplet in triplets:
        subject = triplet.get("subject", "Unknown")
        obj = triplet.get("object", "Unknown")
        predicate = triplet.get("predicate", "unknown")
        subject_type = triplet.get("subject_type", "Condition")
        object_type = triplet.get("object_type", "Condition")

        # Add nodes with type information
        G.add_node(
            subject,
            node_type=subject_type,
            color=NODE_COLORS.get(subject_type, "#CCCCCC"),
        )
        G.add_node(
            obj,
            node_type=object_type,
            color=NODE_COLORS.get(object_type, "#CCCCCC"),
        )

        # Add edge with relationship information
        G.add_edge(
            subject,
            obj,
            label=predicate,
            predicate=predicate,
            weight=EDGE_STYLES.get(predicate, {}).get("width", 1.0),
        )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=150)

    # Use spring layout for better spacing
    pos = nx.spring_layout(
        G,
        k=2,
        iterations=50,
        seed=42,
        scale=3.0,
    )

    # Draw nodes with colors
    node_colors = [G.nodes[node].get("color", "#CCCCCC") for node in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9,
        ax=ax,
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    # Draw edges with different styles
    edge_colors = {}
    for u, v, data in G.edges(data=True):
        predicate = data.get("predicate", "unknown")
        # Use lighter colors for edges
        base_color = NODE_COLORS.get(data.get("predicate", "Condition"), "#AAAAAA")
        edge_colors[(u, v)] = base_color

    # Draw all edges
    for edge_type, style_config in EDGE_STYLES.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("predicate") == edge_type]
        if edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                style=style_config["style"],
                width=style_config["width"],
                alpha=0.7,
                ax=ax,
                edge_color="#666666",
                arrows=True,
                arrowsize=15,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
            )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="darkblue",
        ax=ax,
    )

    # Add legend
    legend_elements = []
    for node_type, color in NODE_COLORS.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, label=node_type))

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        title="Node Types",
        title_fontsize=11,
    )

    # Add title and styling
    title = "Medical Reasoning Graph - Typed Triplet Visualization"
    metadata = data.get("metadata", {})
    if metadata.get("question"):
        title += f"\nQuestion: {metadata.get('question')}"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")

    # Save figure
    try:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"✅ Graph visualization saved to: {output_path}")

        # Print statistics
        print(f"\n📊 Graph Statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Triplets: {len(triplets)}")

        # Count by type
        node_type_counts = {}
        for node, attr in G.nodes(data=True):
            node_type = attr.get("node_type", "Unknown")
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        print(f"\n  Node Types:")
        for node_type, count in sorted(node_type_counts.items()):
            print(f"    {node_type}: {count}")

        plt.close()
        return True
    except Exception as e:
        print(f"❌ Error saving visualization: {str(e)}")
        return False


def main():
    """
    Find latest typed_graph and create visualization
    """
    if not os.path.exists(GRAPH_DIR):
        print(f"❌ Error: Directory '{GRAPH_DIR}' not found")
        print("Run spo_graph.py first to generate reasoning graphs.")
        return False

    # Find latest JSON file
    json_files = [
        os.path.join(GRAPH_DIR, f)
        for f in os.listdir(GRAPH_DIR)
        if f.startswith("typed_graph_") and f.endswith(".json")
    ]

    if not json_files:
        print(f"❌ No typed_graph_*.json files found in {GRAPH_DIR}")
        return False

    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading: {latest_file}\n")

    # Generate a time-distinguished output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(GRAPH_DIR, f"reasoning_graph_{timestamp}.png")

    return visualize_graph(latest_file, output_path)


if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH APPROACH - Enhanced Visualization")
    print("=" * 70 + "\n")

    success = main()
    exit(0 if success else 1)