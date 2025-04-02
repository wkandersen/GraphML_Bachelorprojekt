import pandas as pd
import torch
import os
import sys
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.mini_batches import mini_batches_code
from Packages.loss_function import LossFunction
from Packages.embed_trainer import NodeEmbeddingTrainer
# Set working directory

try:
    data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
    data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
    data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)
except FileNotFoundError:
    os.chdir("..")
    data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
    data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
    data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)

data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

# Extract edges for "paper" -> "cites" -> "paper"
paper_c_paper = data.edge_index_dict[('paper', 'cites', 'paper')]

# Unique paper IDs to keep (Ensure it's a PyTorch tensor)
nums_valid = torch.tensor(data_valid[0])
nums_test = torch.tensor(data_test[0])
nums_train = torch.tensor(data_train[0])

mask_train = torch.isin(paper_c_paper[0], nums_train) | torch.isin(paper_c_paper[1], nums_train)
mask_valid = torch.isin(paper_c_paper[0], nums_valid) | torch.isin(paper_c_paper[1], nums_valid)
mask_test = torch.isin(paper_c_paper[0], nums_test) | torch.isin(paper_c_paper[1], nums_test)

paper_c_paper_train = paper_c_paper.clone()
paper_c_paper_valid = paper_c_paper.clone()
paper_c_paper_test = paper_c_paper.clone()

# Combine the conditions into a single mask that selects only the train edges
mask_train_done = mask_train & ~mask_valid & ~mask_test
mask_valid_done = mask_valid & ~mask_test

# Apply the combined mask to paper_c_paper_train
paper_c_paper_train = paper_c_paper_train[:, mask_train_done]
paper_c_paper_valid = paper_c_paper_valid[:, mask_valid_done]
paper_c_paper_test = paper_c_paper_test[:, mask_test]

#Venues
venues_values = torch.unique(data['y_dict']['paper'])

# Load initial embeddings
embed_venue = torch.load("/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/venue_embeddings.pt")
embed_paper = torch.load("/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/paper_embeddings.pt")

# Initialize dictionaries to store embeddings
paper_dict = copy.deepcopy(embed_paper)  # Ensure we don't modify the original embeddings
venue_dict = copy.deepcopy(embed_venue)
l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# Number of iterations (adjust as needed)
num_iterations = 1 

for i in range(num_iterations):
    print(f"Iteration {i+1}")

    # Generate mini-batches
    mini_b = mini_batches_code(paper_c_paper_train, l_prev, 5, ('paper', 'cites', 'paper'))
    dm, l_next, remapped_datamatrix_tensor = mini_b.node_mapping()

    # Train embeddings and update dictionaries **in place**
    N_emb = NodeEmbeddingTrainer(
        dm=dm,
        remapped_datamatrix_tensor=remapped_datamatrix_tensor,
        paper_dict=paper_dict,  # Pass reference (no copy)
        venue_dict=venue_dict
    )
    paper_dict, venue_dict = N_emb.train()  # Directly update original dictionaries

    # Update node list for the next iteration
    l_prev = l_next

for key in paper_dict:
    paper_dict[key] = paper_dict[key].detach().clone()
    paper_dict[key].requires_grad = False  # Ensure no gradients are tracked

for key in venue_dict:
    venue_dict[key] = venue_dict[key].detach().clone()
    venue_dict[key].requires_grad = False  # Ensure no gradients are tracked

emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))

sample = 1
mini_b_new = mini_batches_code(paper_c_paper_train, list(paper_c_paper_train.unique().numpy()), sample,('paper', 'cites', 'paper'))
dm_new,l_new,remapped_datamatrix_tensor_new = mini_b_new.node_mapping()
new_datamatrix = dm_new[torch.all(dm_new[:, 4:] != 4, dim=1)]
new_remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new[torch.all(remapped_datamatrix_tensor_new[:, 4:] != 4, dim=1)]

loss_function = LossFunction(alpha=1.0, eps=1e-10, use_regularization=True)

new_embedding = torch.nn.Embedding(sample, 2)
print(new_embedding.weight)

new_optimizer = torch.optim.Adam(new_embedding.parameters(), lr=0.01)

venue_dict = venue_dict.copy()
paper_dict = paper_dict.copy()
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    new_optimizer.zero_grad()

    # Concatenate the embeddings
    temp_embed = torch.cat([emb_matrix, new_embedding.weight], dim=0)
    # types = new_datamatrix[:, 3:]
    print(len(temp_embed),len(new_remapped_datamatrix_tensor_new))
    loss = loss_function.compute_loss(temp_embed, new_remapped_datamatrix_tensor_new[:, :3])  # Compute loss
    
    # Backpropagation and optimization
    loss.backward()
    new_optimizer.step()

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")