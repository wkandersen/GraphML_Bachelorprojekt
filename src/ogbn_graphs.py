import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import torch

# Load the dataset
data, _ = torch.load(r"/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/geometric_data_processed.pt")

# Extract citation edges
paper_cites_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
paper_cites_source = paper_cites_edge_index[0].numpy()
paper_cites_target = paper_cites_edge_index[1].numpy()

# Sample 5 papers that actually have citation links
num_samples = 20
valid_paper_pairs = [(src, tgt) for src, tgt in zip(paper_cites_source, paper_cites_target) if src != tgt]
sampled_pairs = random.sample(valid_paper_pairs, num_samples)

# Extract unique paper IDs from sampled pairs
sampled_papers = set()
for src, tgt in sampled_pairs:
    sampled_papers.add(src)
    sampled_papers.add(tgt)

# Create a bipartite graph
B = nx.Graph()

# Create separate nodes for left and right
left_nodes = {p: f"L_{p}" for p in sampled_papers}  # Left-side nodes
right_nodes = {p: f"R_{p}" for p in sampled_papers}  # Right-side nodes

B.add_nodes_from(left_nodes.values(), bipartite=0)  # Left column
B.add_nodes_from(right_nodes.values(), bipartite=1)  # Right column

# Add edges between left and right based on sampled citation relationships
for src, tgt in sampled_pairs:
    B.add_edge(left_nodes[src], right_nodes[tgt])  # Edge only between columns

# ----------------------------
# Step 2: Generate Random Walks for Embedding
# ----------------------------

def generate_metapath_walks(G, nodes, walk_length=10, num_walks=10):
    """Generate random walks across bipartite graph (switching node sets)."""
    walks = []
    for _ in range(num_walks):
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.append(walk)
    return walks

# Generate walks on the bipartite graph
all_nodes = list(B.nodes)
walks = generate_metapath_walks(B, all_nodes, walk_length=5, num_walks=10)

# ----------------------------
# Step 3: Train Word2Vec Embeddings
# ----------------------------
embedding_dim = 8
model = Word2Vec(walks, vector_size=embedding_dim, window=3, min_count=0, sg=1, workers=1, epochs=100)
embeddings = {node: model.wv[node] for node in all_nodes}

# Convert embeddings to tensor
emb_matrix = torch.tensor(np.array([embeddings[node] for node in all_nodes]), dtype=torch.float)

# ----------------------------
# Step 4: Compute Probabilistic Edge Likelihood
# ----------------------------
def edge_probability(z_i, z_j, alpha=1.0):
    """Compute the probability of an edge given embeddings."""
    dist = np.linalg.norm(z_i - z_j) ** 2  # Squared Euclidean distance
    return 1 / (1 + np.exp(alpha * dist))  # Logistic function

# ----------------------------
# Step 5: Reconstruct a New Bipartite Graph Using Probabilities
# ----------------------------
new_G = nx.Graph()
new_G.add_nodes_from(all_nodes)

alpha = 5  # Scaling factor for probability (tuneable)

# Ensure edges are only between left and right sets
for left in left_nodes.values():
    for right in right_nodes.values():
        prob = edge_probability(embeddings[left], embeddings[right], alpha)
        if random.random() < prob:  # Sample based on probability
            new_G.add_edge(left, right)

# ----------------------------
# Step 6: Visualize Original and Reconstructed Graphs
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Create layout positions
pos = {}
for i, paper in enumerate(sampled_papers):
    pos[left_nodes[paper]] = (0, i)  # Left column
    pos[right_nodes[paper]] = (1, i)  # Right column

# Plot Original Bipartite Graph
nx.draw(B, pos, with_labels=True, node_size=500, edge_color='gray', node_color='skyblue', alpha=0.7, font_size=8, ax=axes[0])
axes[0].set_title("Original Bipartite Graph")

# Plot Embeddings (2D Visualization)
emb_array = emb_matrix.detach().numpy()
for idx, node in enumerate(all_nodes):
    color = 'skyblue' if node in left_nodes.values() else 'salmon'
    axes[1].scatter(emb_array[idx, 0], emb_array[idx, 1], c=color, s=150)
    axes[1].text(emb_array[idx, 0], emb_array[idx, 1], node, fontsize=8, ha='center', va='center')
axes[1].set_title("Learned Node Embeddings (Latent Space)")

# Plot Reconstructed Graph
nx.draw(new_G, pos, with_labels=True, node_size=500, edge_color='gray', node_color='lightblue', alpha=0.7, font_size=8, ax=axes[2])
axes[2].set_title("Reconstructed Graph (Edges Only Between Columns)")

plt.show()

# ----------------------------
# Step 7: Evaluating the reconstructed Graph
# ----------------------------

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