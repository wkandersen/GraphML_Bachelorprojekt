# Re-import necessary libraries after execution reset
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances
import torch

data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt")

paper_cites_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]

random.seed(99)

# ----------------------------
# Step 1: Create a Bipartite Graph from Dataset
# ----------------------------

# Sample 5 random papers
num_samples = 5
paper_ids = torch.unique(paper_cites_edge_index)  # Get all unique paper IDs
random_papers = random.sample(list(paper_ids), num_samples)

# Construct the bipartite graph using the sampled papers
citing_papers = []
cited_papers = []

# Find which papers cite which ones
for paper in random_papers:
    # Get all papers citing the current paper
    citations = (paper_cites_edge_index[0] == paper).nonzero(as_tuple=True)[0]
    
    # Append to citing and cited lists (citing paper -> cited paper)
    citing_papers.extend([paper] * len(citations))
    cited_papers.extend(paper_cites_edge_index[1][citations].tolist())

# Convert to tensors
citing_papers = torch.tensor(citing_papers)
cited_papers = torch.tensor(cited_papers)

# Create a NetworkX graph for visualization
G = nx.Graph()

# Add edges to the graph
for i in range(citing_papers.shape[0]):
    G.add_edge(citing_papers[i].item(), cited_papers[i].item())

# Get the set of nodes for each group
nodes_set_1 = set(citing_papers.tolist())  # Citing papers
nodes_set_2 = set(cited_papers.tolist())   # Cited papers

# ----------------------------
# Step 2: Generate Random Walks for Embedding
# ----------------------------
def generate_metapath_walks(G, nodes, walk_length=10, num_walks=10):
    """Generate metapath2vec-style random walks (bipartite alternating)."""
    walks = []
    for _ in range(num_walks):
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                if current in nodes_set_1:
                    valid_neighbors = [n for n in neighbors if n in nodes_set_2]
                else:
                    valid_neighbors = [n for n in neighbors if n in nodes_set_1]
                if not valid_neighbors:
                    break
                current = random.choice(valid_neighbors)
                walk.append(current)
            walks.append(walk)
    return walks

# Get all unique nodes from the graph for random walks
all_nodes = list(G.nodes)

# Generate random walks from the bipartite graph
walks = generate_metapath_walks(G, all_nodes, walk_length=5, num_walks=10)

# hyperparameters walk_length and num_walks

# ----------------------------
# Step 3: Train Word2Vec Embeddings
# ----------------------------
embedding_dim = 8 # hyperparameter
model = Word2Vec(walks, vector_size=embedding_dim, window=3, min_count=0, sg=1, workers=1, epochs=100)
embeddings = {node: model.wv[node] for node in all_nodes}
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

alpha = 0.5  # Scaling factor for probability (tuneable) - hyperparameter

# Add edges based on the learned embeddings
for u in all_nodes:
    for v in all_nodes:
        if u != v:
            prob = edge_probability(embeddings[u], embeddings[v], alpha)
            if random.random() < prob:  # Sample based on probability
                new_G.add_edge(u, v)

# ----------------------------
# Step 6: Visualize Original and Reconstructed Graphs
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Create positions for plotting
pos = {node: (0, idx) for idx, node in enumerate(citing_papers.tolist())}
pos.update({node: (1, idx) for idx, node in enumerate(cited_papers.tolist())})

# Plot Original Graph
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", edge_color="gray", ax=axes[0])
axes[0].set_title("Original Bipartite Graph")

# Plot Embeddings (2D Visualization)
emb_array = emb_matrix.detach().numpy()
for idx, node in enumerate(all_nodes):
    color = 'skyblue' if node in citing_papers.tolist() else 'salmon'
    axes[1].scatter(emb_array[idx, 0], emb_array[idx, 1], c=color, s=150)
    axes[1].text(emb_array[idx, 0], emb_array[idx, 1], node, fontsize=12, ha='center', va='center')
axes[1].set_title("Learned Node Embeddings (Latent Space)")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")

# Plot Reconstructed Graph
nx.draw(new_G, pos, with_labels=True, node_size=1000, node_color="lightblue", edge_color="gray", ax=axes[2])
axes[2].set_title("Reconstructed Graph from Probabilities")

plt.show()
