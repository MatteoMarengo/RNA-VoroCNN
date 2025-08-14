from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.drawing import rna_draw
import matplotlib.pyplot as plt
import pickle
import json
import networkx as nx

# Step 1: Convert CIF to 2.5D graph
print("Converting 6pq7.cif to 2.5D graph...")
G = fr3d_to_graph("6pq7.cif")

# Step 2: Display graph information
print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"Graph attributes: {G.graph}")

# Step 3: Create and save visualization
print("Creating visualization...")
fig, ax = plt.subplots(figsize=(12, 10))

# Draw the RNA graph
rna_draw(G, show=False, layout="spring", ax=ax)

# Add title
ax.set_title("6pq7 - 2.5D RNA Graph", fontsize=16, fontweight='bold')

# Save the visualization
output_file = "6pq7_2.5D_graph.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved as: {output_file}")

# Step 4: Save the graph object in multiple formats
print("Saving graph object...")

# Save as pickle (preserves all Python objects and attributes)
with open("6pq7_2.5D_graph.pkl", "wb") as f:
    pickle.dump(G, f)
print("✓ Graph saved as pickle: 6pq7_2.5D_graph.pkl")

# Save as GraphML (with cleaned attributes for compatibility)
try:
    # Create a copy of the graph with cleaned attributes
    G_clean = G.copy()
    
    # Clean node attributes (remove lists and complex objects)
    for node in G_clean.nodes():
        node_attrs = G_clean.nodes[node].copy()
        cleaned_attrs = {}
        for key, value in node_attrs.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_attrs[key] = value
            elif isinstance(value, list):
                cleaned_attrs[key] = str(value)  # Convert lists to strings
            else:
                cleaned_attrs[key] = str(value)  # Convert other types to strings
        G_clean.nodes[node].clear()
        G_clean.nodes[node].update(cleaned_attrs)
    
    # Clean edge attributes
    for edge in G_clean.edges():
        edge_attrs = G_clean.edges[edge].copy()
        cleaned_attrs = {}
        for key, value in edge_attrs.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_attrs[key] = value
            elif isinstance(value, list):
                cleaned_attrs[key] = str(value)
            else:
                cleaned_attrs[key] = str(value)
        G_clean.edges[edge].clear()
        G_clean.edges[edge].update(cleaned_attrs)
    
    # Clean graph attributes
    graph_attrs = G_clean.graph.copy()
    cleaned_graph_attrs = {}
    for key, value in graph_attrs.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned_graph_attrs[key] = value
        elif isinstance(value, list):
            cleaned_graph_attrs[key] = str(value)
        else:
            cleaned_graph_attrs[key] = str(value)
    G_clean.graph.clear()
    G_clean.graph.update(cleaned_graph_attrs)
    
    nx.write_graphml(G_clean, "6pq7_2.5D_graph.graphml")
    print("✓ Graph saved as GraphML: 6pq7_2.5D_graph.graphml")
    
except Exception as e:
    print(f"⚠ Could not save as GraphML: {e}")
    print("  (This is normal for RNA graphs with complex attributes)")

# Save as JSON (human-readable, but may lose some complex attributes)
try:
    graph_data = nx.node_link_data(G)
    with open("6pq7_2.5D_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)
    print("✓ Graph saved as JSON: 6pq7_2.5D_graph.json")
except Exception as e:
    print(f"⚠ Could not save as JSON: {e}")

print("All files saved successfully!")