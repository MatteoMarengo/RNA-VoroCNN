# RNA-VoroCNN

In the original paper, VoroCNN is described as a method specifically designed for analyzing protein structures using a 3D Voronoi tesselation.

How VoroCNN Works for Proteins:

1/ Input: A 3D atomic model of a protein

2/ 3D Voronoi Tesselation:
The method uses a tool called Voronota to perform a 3D Voronoi tessellation of the space around all the atoms in the protein. This partitions the space into polyhedral cells, where
each cell contains one atom and represents the region of space that is closer to that atom than to any other.

3/ Graph Construction: The tesselation is converted into a graph
- Nodes: each atom is a node
- Edges: an edge is created between two nodes (atoms) if their corresponding Voronoi cells share a face. This means the atoms are in direct contact or very close proximity.

4/ Node Features:
Each node (atom) is designed a feature vector. In the paper, this is a one-hot encoded vector representing the atom type (e.g., C, N, O, S) with 167 possible types.

5/ Graph Convolutional Network (GCN): This graph, with its nodes, edges, and features, is then fed into a Graph Convolutional Network.

The GCN learns to predict a property (in this case, the local quality of the protein model) by analyzing the local network of atoms and their types.