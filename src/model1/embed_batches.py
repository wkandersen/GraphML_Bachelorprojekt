# import torch
# import copy
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from Packages.mini_batches import mini_batches_code
# from Packages.embed_trainer import NodeEmbeddingTrainer
# from Packages.data_divide import paper_c_paper_train
# import gc

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# print("starting")
# # Load initial embeddings
# embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device)
# embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device)

# # Initialize dictionaries to store embeddings
# paper_dict = {k: v.clone() for k, v in embed_paper.items()}
# venue_dict = {k: v.clone() for k, v in embed_venue.items()}

# l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes



# # hyperparameters
# batch_size = 200
# num_epochs = 10
# lr = 0.01
# alpha = 1
# lam = 0.01
# # num_iterations =  int(len(paper_dict)/batch_size)-1 # we need to be able to look at the complete dataset
# num_iterations = 100

# for i in range(num_iterations):
#     print(f"Iteration {i+1}")

#     # Generate mini-batches
#     mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'))
#     dm, l_next, remapped_datamatrix_tensor,random_sample = mini_b.node_mapping()

#     # Move data to GPU
#     dm = dm.to(device)
#     remapped_datamatrix_tensor = remapped_datamatrix_tensor.to(device)

#     # Train embeddings and update dictionaries **in place**
#     N_emb = NodeEmbeddingTrainer(
#         dm=dm,
#         remapped_datamatrix_tensor=remapped_datamatrix_tensor,
#         paper_dict=paper_dict,  # Pass reference (no copy)
#         venue_dict=venue_dict,
#         num_epochs=num_epochs,
#         lr=lr,
#         alpha=alpha,
#         lam=lam
#     )
#     paper_dict, venue_dict,loss = N_emb.train()  # Directly update original dictionaries

#     # Update node list for the next iteration
#     l_prev = l_next

#     # Cleanup
#     if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
#         import gc
#         gc.collect()
#         torch.cuda.empty_cache()

# for key in paper_dict:
#     paper_dict[key] = paper_dict[key].detach().clone().cpu()
#     paper_dict[key].requires_grad = False  # Ensure no gradients are tracked

# for key in venue_dict:
#     venue_dict[key] = venue_dict[key].detach().clone().cpu()
#     venue_dict[key].requires_grad = False  # Ensure no gradients are tracked

# torch.save(paper_dict, "dataset/ogbn_mag/processed/hpc/paper_dict.pt")
# torch.save(venue_dict, "dataset/ogbn_mag/processed/hpc/venue_dict.pt")

# emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))

# torch.save(emb_matrix, "dataset/ogbn_mag/processed/hpc/emb_matrix.pt")

# print('Embed_batches done')



import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import LossFunction  # Assuming this is defined here
from Packages.data_divide import paper_c_paper_train
import gc
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Amount of devices: {torch.cuda.device_count()}")

print("Starting")

# Load initial embeddings
embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device,mmap=True)
embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device,mmap=True)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)


# Initialize dictionaries to store embeddings
paper_dict = {k: v.clone() for k, v in embed_paper.items()}
venue_dict = {k: v.clone() for k, v in embed_venue.items()}

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# Hyperparameters
batch_size = 250
num_epochs = 10
lr = 0.01
alpha = 1
lam = 0.01
num_iterations = 10

# Define a modular embedding model
class EmbeddingModel(nn.Module):
    def __init__(self, num_papers, num_venues, embedding_dim):
        super().__init__()
        self.papernode_embeddings = nn.Embedding(num_papers, embedding_dim)
        self.venuenode_embeddings = nn.Embedding(num_venues, embedding_dim)

    def forward(self):
        return torch.cat((self.papernode_embeddings.weight, self.venuenode_embeddings.weight), dim=0)

# Loss function 
loss_function = LossFunction(alpha=alpha, eps=1e-10, use_regularization=True, lam=lam)

for i in range(num_iterations):
    print(f"Iteration {i+1}")

    # Generate mini-batches
    mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'),data)
    dm, l_next, remapped_datamatrix_tensor, random_sample = mini_b.node_mapping()

    # Split paper and venue indices
    dm1 = dm[dm[:, 4] != 4]
    dm2 = dm[dm[:, 4] == 4]
    paper_indices = torch.cat([torch.unique(dm1[:, 1]), torch.unique(dm1[:, 2])], dim=0)
    venue_indices = torch.unique(dm2[:, 2], dim=0)

    # Create model
    model = EmbeddingModel(len(paper_indices), len(venue_indices), embedding_dim=2)

    #Wrap in DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to GPU
    remapped_datamatrix_tensor = remapped_datamatrix_tensor.to(device)

    # Training
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z = model()
        loss = loss_function.compute_loss(z, remapped_datamatrix_tensor[:, :3])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    #Extract weights from model
    z = model().cpu().detach()
    paper_weights = z[:len(paper_indices)]
    venue_weights = z[len(paper_indices):]

    for idx, node in enumerate(paper_indices):
        paper_dict[int(node)] = paper_weights[idx].clone()

    for idx, node in enumerate(venue_indices):
        venue_dict[int(node)] = venue_weights[idx].clone()

    # Update node list for the next iteration
    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

# Finalize: detach and move to CPU
for key in paper_dict:
    paper_dict[key] = paper_dict[key].detach().clone().cpu()
    paper_dict[key].requires_grad = False

for key in venue_dict:
    venue_dict[key] = venue_dict[key].detach().clone().cpu()
    venue_dict[key].requires_grad = False

# Save
torch.save(paper_dict, "dataset/ogbn_mag/processed/hpc/paper_dict.pt")
torch.save(venue_dict, "dataset/ogbn_mag/processed/hpc/venue_dict.pt")

emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))
torch.save(emb_matrix, "dataset/ogbn_mag/processed/hpc/emb_matrix.pt")

print('Embed_batches done')

