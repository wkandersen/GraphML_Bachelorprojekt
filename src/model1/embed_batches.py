import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train

print("starting")
# Load initial embeddings
embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt")
embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt")

# Initialize dictionaries to store embeddings
paper_dict = {k: v.clone() for k, v in embed_paper.items()}
venue_dict = {k: v.clone() for k, v in embed_venue.items()}

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# Number of iterations (adjust as needed)
num_iterations =  622

for i in range(num_iterations):
    print(f"Iteration {i+1}")

    # Generate mini-batches
    mini_b = mini_batches_code(paper_c_paper_train, l_prev, 1000, ('paper', 'cites', 'paper'))
    dm, l_next, remapped_datamatrix_tensor,random_sample = mini_b.node_mapping()

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

torch.save(paper_dict, "dataset/ogbn_mag/processed/hpc/paper_dict.pt")
torch.save(venue_dict, "dataset/ogbn_mag/processed/hpc/venue_dict.pt")

emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))

torch.save(emb_matrix, "dataset/ogbn_mag/processed/hpc/emb_matrix.pt")

print('Embed_batches done')
