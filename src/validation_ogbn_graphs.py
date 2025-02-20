import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import graph_edit_distance
from ogbn_graphs import B, new_G

def compare_bipartite_graphs(G1, G2):
    """
    Compare and evaluate differences between two bipartite graphs.
    """
    # Ensure graphs are bipartite
    if not nx.is_bipartite(G1) or not nx.is_bipartite(G2):
        raise ValueError("One or both graphs are not bipartite.")

    # Compute basic properties
    print("Graph Properties:")
    print(f"Original: Nodes = {G1.number_of_nodes()}, Edges = {G1.number_of_edges()}")
    print(f"Generated: Nodes = {G2.number_of_nodes()}, Edges = {G2.number_of_edges()}")

    # Compute edge overlap
    G1_edges = set(G1.edges())
    G2_edges = set(G2.edges())
    common_edges = G1_edges.intersection(G2_edges)
    jaccard_edges = len(common_edges) / len(G1_edges.union(G2_edges))

    print(f"Common Edges: {len(common_edges)}")
    print(f"Jaccard Similarity of Edges: {jaccard_edges:.4f}")

# Compare graphs
compare_bipartite_graphs(B, new_G)
