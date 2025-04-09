import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train
import gc
import wandb

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
# Load initial embeddings
embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device)
embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device)

# Initialize dictionaries to store embeddings
paper_dict = {k: v.clone() for k, v in embed_paper.items()}
venue_dict = {k: v.clone() for k, v in embed_venue.items()}

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# hyperparameters
batch_size = 3
num_epochs = 10
lr = 0.01
alpha = 0.5
lam = 0.01
num_iterations =  5 # we need to be able to look at the complete dataset

run = wandb.init(
    project="Bachelor_projekt",
    name="run_${now:%Y-%m-%d %H:%M:%S}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "alpha": alpha,
        "lam": lam
    },
)
for i in range(num_iterations):
    print(f"Iteration {i+1}")

    # Generate mini-batches
    mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'))
    dm, l_next, remapped_datamatrix_tensor,random_sample = mini_b.node_mapping()

    # Move data to GPU
    dm = dm.to(device)
    remapped_datamatrix_tensor = remapped_datamatrix_tensor.to(device)

    # Train embeddings and update dictionaries **in place**
    N_emb = NodeEmbeddingTrainer(
        dm=dm,
        remapped_datamatrix_tensor=remapped_datamatrix_tensor,
        paper_dict=paper_dict,  # Pass reference (no copy)
        venue_dict=venue_dict,
        num_epochs=num_epochs,
        lr=lr,
        alpha=alpha,
        lam=lam
    )
    paper_dict, venue_dict,loss = N_emb.train()  # Directly update original dictionaries

    wandb.log({"loss": loss, "iteration": i+1})

    # Update node list for the next iteration
    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

for key in paper_dict:
    paper_dict[key] = paper_dict[key].detach().clone().cpu()
    paper_dict[key].requires_grad = False  # Ensure no gradients are tracked

for key in venue_dict:
    venue_dict[key] = venue_dict[key].detach().clone().cpu()
    venue_dict[key].requires_grad = False  # Ensure no gradients are tracked

torch.save(paper_dict, "dataset/ogbn_mag/processed/hpc/paper_dict.pt")
torch.save(venue_dict, "dataset/ogbn_mag/processed/hpc/venue_dict.pt")

emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))

torch.save(emb_matrix, "dataset/ogbn_mag/processed/hpc/emb_matrix.pt")

print('Embed_batches done')
