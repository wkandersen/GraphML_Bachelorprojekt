import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train
from Packages.prediction import Prediction
from embed_valid_sample import EmbeddingTrainer_valid
import gc
import wandb
from datetime import datetime

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
embedding_dim = 2
# Load initial embeddings
embed_venue = torch.load(f"dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device)
embed_paper = torch.load(f"dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device)
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device, weights_only=False)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

# Initialize dictionaries to store embeddings
paper_dict = {k: v.clone() for k, v in embed_paper.items()}
venue_dict = {k: v.clone() for k, v in embed_venue.items()}

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# hyperparameters
batch_size = 5
num_epochs = 10
lr = 0.01
alpha = 0.1
lam = 0.001
num_iterations =  int(len(paper_dict)/batch_size)-1 # we need to be able to look at the complete dataset
saved_checkpoints = []
max_saved = 2
save_every_iter = 5

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
    dm = dm.to(device)
    remapped_datamatrix_tensor = remapped_datamatrix_tensor.to(device)

    # Train embeddings and update dictionaries *in place*
    N_emb = NodeEmbeddingTrainer(
        dm=dm,
        remapped_datamatrix_tensor=remapped_datamatrix_tensor,
        paper_dict=paper_dict,  # Pass reference (no copy)
        venue_dict=venue_dict,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        lr=lr,
        alpha=alpha,
        lam=lam,
        device=device
    )

    paper_dict, venue_dict, loss = N_emb.train()  # Directly update original dictionaries

    if i % 100 == 0:
        emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))
        trainer_valid = EmbeddingTrainer_valid(emb_matrix=emb_matrix,embedding_dim=embedding_dim,num_epochs=50,samples=300,learning_rate=lr,alpha=alpha)
        valid_dict,loss_valid = trainer_valid.train()

        pred_class = Prediction(alpha=alpha, paper_dict=paper_dict, valid_dict=valid_dict, venue_dict=venue_dict, venue_value=venue_value)
        acc_train, acc_valid = pred_class.accuracy()
        wandb.log({"loss_valid": loss_valid, "iteration": i+1})
        wandb.log({"acc_valid": acc_valid, "iteration": i+1})
        wandb.log({"acc_train": acc_train, "iteration": i+1})

    wandb.log({"loss": loss, "iteration": i+1})

    # Update node list for the next iteration
    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if (i + 1) % save_every_iter == 0:
        iter_id = i + 1

        os.makedirs("checkpoint", exist_ok=True)
        trainer_path = f"checkpoint/trainer_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"
        paper_path = f"checkpoint/paper_dict_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"
        venue_path = f"checkpoint/venue_dict_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"
        l_prev_path = f"checkpoint/l_prev_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"

        N_emb.save_checkpoint(trainer_path)
        torch.save(paper_dict, paper_path)
        torch.save(venue_dict, venue_path)
        torch.save(l_prev, l_prev_path)

        saved_checkpoints.append((trainer_path, paper_path, venue_path, l_prev_path))

        # Remove older checkpoints if more than 2 are saved
        if len(saved_checkpoints) > max_saved:
            old_files = saved_checkpoints.pop(0)
            for f in old_files:
                if os.path.exists(f):
                    os.remove(f)
        
        # # Log artifact to wandb (optional, still useful)
        # artifact = wandb.Artifact(f"embedding_checkpoint_{iter_id}", type="model")
        # artifact.add_file(trainer_path)
        # artifact.add_file(paper_path)
        # artifact.add_file(venue_path)
        # wandb.log_artifact(artifact)


for key in paper_dict:
    paper_dict[key] = paper_dict[key].detach().clone().cpu()
    paper_dict[key].requires_grad = False  # Ensure no gradients are tracked

for key in venue_dict:
    venue_dict[key] = venue_dict[key].detach().clone().cpu()
    venue_dict[key].requires_grad = False  # Ensure no gradients are tracked

torch.save(paper_dict, f"dataset/ogbn_mag/processed/hpc/paper_dict_{embedding_dim}_{num_epochs}_epoch.pt")
torch.save(venue_dict, f"dataset/ogbn_mag/processed/hpc/venue_dict_{embedding_dim}_{num_epochs}_epoch.pt")

emb_matrix = torch.stack(list(paper_dict.values()) + list(venue_dict.values()))

torch.save(emb_matrix, f"dataset/ogbn_mag/processed/hpc/emb_matrix_{embedding_dim}_{num_epochs}_epoch.pt")

print('Embed_batches done')