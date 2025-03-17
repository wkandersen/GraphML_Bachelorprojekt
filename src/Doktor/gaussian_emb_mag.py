import networkx as nx
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.optim as optim
import seaborn as sns

# Load the dataset
data, _ = torch.load(r"/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/geometric_data_processed.pt")

# Extract citation edges
paper_cites_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
paper_cites_source = paper_cites_edge_index[0].numpy()
paper_cites_target = paper_cites_edge_index[1].numpy()

# Sample 5 papers that actually have citation links
num_samples = 50
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

all_nodes = list(B.nodes)
edges = set(B.edges())

# 2️ Create the Data Matrix (Edge / Non-Edge Pairs)
datamatrix = []
for u in left_nodes:
    for v in right_nodes:
        label = 1 if (left_nodes[u], right_nodes[v]) in edges else 0  # Check edge existence
        u_idx = all_nodes.index(left_nodes[u])  # Convert node labels to indices
        v_idx = all_nodes.index(right_nodes[v])  # Convert node labels to indices
        datamatrix.append([label, u_idx, v_idx])  # Store indices instead of labels

datamatrix = np.array(datamatrix)

# Convert datamatrix to a PyTorch tensor
datamatrix_tensor = torch.tensor(datamatrix, dtype=torch.long)

# 3 Create intialization of embedding using Gaussian distribution
def initialize_embeddings_gaussian(num_nodes, embedding_dim):
    embeddings = {}
    sigma = 1.0 / math.sqrt(embedding_dim)
    for node in range(num_nodes):
        embeddings[node] = [random.gauss(0, sigma) for _ in range(embedding_dim)]
    return embeddings

num_nodes = len(all_nodes)
embedding_dim = 6
embeddings = initialize_embeddings_gaussian(num_nodes,embedding_dim)

embedding_tensor = torch.tensor(
    [embeddings[node] for node in range(num_nodes)],
    dtype=torch.float32, requires_grad=True
)

# 4 create functions for edge_probability and loss

def edge_probability(z_i, z_j, alpha=1.0):
    """Compute the probability of an edge given embeddings."""
    dist = torch.norm(z_i - z_j) ** 2  # Squared Euclidean distance using PyTorch
    return 1 / (1 + torch.exp(-alpha + dist))  # Logistic function using PyTorch

def link_prediction_loss(z, datamatrix_tensor, alpha=1.0, eps=1e-8):
    sum_loss = 0
    for entry in datamatrix_tensor:
        label, u_idx, v_idx = entry
        z_u = z[u_idx]
        z_v = z[v_idx]
        prob = edge_probability(z_u, z_v, alpha)

        # Numerical stability: clamp probabilities to avoid log(0)
        prob = torch.clamp(prob, eps, 1 - eps)

        sum_loss += label * torch.log(prob) + (1 - label) * torch.log(1 - prob)

    return -sum_loss / len(datamatrix_tensor)

# 5 optimize the embeddings

optimizer = optim.Adam([embedding_tensor],lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = link_prediction_loss(embedding_tensor,datamatrix_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}], Loss: {loss.item():.4f}")

emb_matrix = embedding_tensor.detach().numpy()

# 6 Reconstruct Graph Using Edge Probabilities
new_B = nx.Graph()
new_B.add_nodes_from(left_nodes, bipartite=0)
new_B.add_nodes_from(right_nodes, bipartite=1)

threshold = 0.5  # Edge threshold
prob_list = []
with torch.no_grad():
    for u in range(len(all_nodes)):
        for v in range(u + 1, len(all_nodes)):
            if (u in range(len(left_nodes)) and v in range(len(left_nodes), len(all_nodes))) or \
               (v in range(len(left_nodes)) and u in range(len(left_nodes), len(all_nodes))): 
                
                # Convert numpy embeddings back to PyTorch tensors for edge_probability calculation
                u_embedding = torch.tensor(emb_matrix[u], dtype=torch.float32)
                v_embedding = torch.tensor(emb_matrix[v], dtype=torch.float32)
                
                # Compute edge probability
                prob = edge_probability(u_embedding, v_embedding, alpha=1.0)
                prob_list.append(prob.item())
                
                # Add edge to new graph if probability is above threshold
                if prob > threshold:
                    new_B.add_edge(all_nodes[u], all_nodes[v])

# 7 Calculate Evaluation Metrics and compute confusion metrics
# Initialize counters
TP = FP = TN = FN = 0

# Check for all node pairs
with torch.no_grad():
    for u in range(len(all_nodes)):
        for v in range(u + 1, len(all_nodes)):
            if (u in range(len(left_nodes)) and v in range(len(left_nodes), len(all_nodes))) or \
               (v in range(len(left_nodes)) and u in range(len(left_nodes), len(all_nodes))):
                
                # Determine the ground truth (actual edge in original graph)
                label = 1 if (all_nodes[u], all_nodes[v]) in edges or (all_nodes[v], all_nodes[u]) in edges else 0
                
                # Compute edge probability
                u_embedding = torch.tensor(emb_matrix[u], dtype=torch.float32)
                v_embedding = torch.tensor(emb_matrix[v], dtype=torch.float32)
                prob = edge_probability(u_embedding, v_embedding, alpha=1.0)
                
                # Prediction (thresholding at 50%)
                predicted_label = 1 if prob > threshold else 0
                
                # Update counters based on true and predicted labels
                if label == 1 and predicted_label == 1:
                    TP += 1
                elif label == 0 and predicted_label == 1:
                    FP += 1
                elif label == 0 and predicted_label == 0:
                    TN += 1
                elif label == 1 and predicted_label == 0:
                    FN += 1

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Calculate Recall (Sensitivity)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# Calculate Precision (Positive Predictive Value)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Output results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall (Sensitivity): {recall * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"F1-score: {f1_score * 100:.2f}%")

# Define the confusion matrix
conf_matrix = np.array([[TN, FP], 
                        [FN, TP]])

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# 8 Plot Original Graph, Embeddings, and Reconstructed Graph
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Create layout positions for the original graph manually from `B`
# This assumes the graph `B` has nodes that correspond to the sampled papers
# So the positions will be manually defined to ensure both graphs have the same positions
pos = {}
for i, paper in enumerate(sampled_papers):
    pos[left_nodes[paper]] = (0, i)  # Left column
    pos[right_nodes[paper]] = (1, i)  # Right column

# Ensure that all nodes in new_B also have positions in `pos`
# For the reconstructed graph `new_B`, only the nodes present in `pos` are valid
# Loop through all nodes in `new_B` and add them to `pos` if they're not already there
for node in new_B.nodes():
    if node not in pos:
        # Define position for the node if it doesn't already have one
        # You can create a default position here, for instance:
        pos[node] = (np.random.random(), np.random.random())  # Random position

# Plot Original Bipartite Graph (Using same positions)
nx.draw(B, pos, with_labels=True, node_size=500, edge_color='gray', node_color='skyblue', alpha=0.7, font_size=8, ax=axes[0])
axes[0].set_title("Original Bipartite Graph")

# Plot Embeddings (2D Visualization)
emb_array = emb_matrix
for idx, node in enumerate(all_nodes):
    color = 'skyblue' if node in left_nodes.values() else 'salmon'
    axes[1].scatter(emb_array[idx, 0], emb_array[idx, 1], c=color, s=150)
    axes[1].text(emb_array[idx, 0], emb_array[idx, 1], node, fontsize=8, ha='center', va='center')
axes[1].set_title("Learned Node Embeddings (Latent Space)")

# Plot Reconstructed Graph with the same positions as the original graph
nx.draw(new_B, pos, with_labels=True, node_size=500, edge_color='gray', node_color='skyblue', alpha=0.7, font_size=8, ax=axes[2])
axes[2].set_title("Reconstructed Graph (Edges Only Between Columns)")

plt.show()
