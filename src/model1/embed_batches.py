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
import wandb
from datetime import datetime

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Amount of devices: {torch.cuda.device_count()}")

print("Starting")

# Load initial embeddings
embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device)
embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

# Initialize dictionaries to store embeddings
paper_dict = {k: v.clone() for k, v in embed_paper.items()}
venue_dict = {k: v.clone() for k, v in embed_venue.items()}

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# hyperparameters
batch_size = 3
num_epochs = 4
lr = 0.01
alpha = 0.5
lam = 0.01
num_iterations =  6 # we need to be able to look at the complete dataset
saved_checkpoints = []
max_saved = 2
save_every_it = 2

run = wandb.init(
    project="Bachelor_projekt",
    name=f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}",
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
    mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'),data)
    dm, l_next, remapped_datamatrix_tensor,random_sample = mini_b.node_mapping()

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

    wandb.log({"loss": loss, "iteration": i+1})

    # Update node list for the next iteration
    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

    if (i + 1) % save_every_it == 0:
        iter_id = i + 1

        os.makedirs("checkpoint", exist_ok=True)
        # Define filenames with iteration
        trainer_path = f"checkpoint/trainer_iter{iter_id}.pt"
        paper_path = f"checkpoint/paper_dict_iter{iter_id}.pt"
        venue_path = f"checkpoint/venue_dict_iter{iter_id}.pt"

        # Save current versions
        N_emb.save_checkpoint(trainer_path)
        torch.save(paper_dict, paper_path)
        torch.save(venue_dict, venue_path)

        # Add current to the list
        saved_checkpoints.append((trainer_path, paper_path, venue_path))

        # Remove older ones if more than 2 are saved
        if len(saved_checkpoints) > max_saved:
            old_trainer, old_paper, old_venue = saved_checkpoints.pop(0)
            for f in [old_trainer, old_paper, old_venue]:
                if os.path.exists(f):
                    os.remove(f)

        # Log artifact to wandb (optional, still useful)
        artifact = wandb.Artifact(f"embedding_checkpoint_{iter_id}", type="model")
        artifact.add_file(trainer_path)
        artifact.add_file(paper_path)
        artifact.add_file(venue_path)
        wandb.log_artifact(artifact)


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

