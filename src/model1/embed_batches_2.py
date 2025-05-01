import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train
import gc
import wandb
from datetime import datetime

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
embedding_dim = 2
# Load initial embeddings
embed_dict = torch.load(f"dataset/ogbn_mag/processed/hpc/collected_embeddings_2.pt", map_location=device)
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device, weights_only=False)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

# hyperparameters
batch_size = 5
num_epochs = 10
lr = 0.01
alpha = 0.1
lam = 0.001
num_iterations =  int(len(embed_dict)/batch_size)-1 # we need to be able to look at the complete dataset
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
    dm, l_next,random_sample = mini_b.datamatrix()

    # Move data to GPU
    dm = dm.to(device)

    # Train embeddings and update dictionaries *in place*
    N_emb = NodeEmbeddingTrainer(
        dm=dm,
        embed_dict=embed_dict,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        lr=lr,
        alpha=alpha,
        lam=lam,
        device=device
    )

    embed_dict, loss = N_emb.train()  # Directly update original dictionaries

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
        embed_path = f"checkpoint/embed_dict_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"
        l_prev_path = f"checkpoint/l_prev_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"

        N_emb.save_checkpoint(trainer_path)
        torch.save(embed_dict, embed_path)
        torch.save(l_prev, l_prev_path)

        saved_checkpoints.append((trainer_path, embed_path, l_prev_path))

        # Remove older checkpoints if more than 2 are saved
        if len(saved_checkpoints) > max_saved:
            old_files = saved_checkpoints.pop(0)
            for f in old_files:
                if os.path.exists(f):
                    os.remove(f)


for key in embed_dict:
    embed_dict[key] = embed_dict[key].detach().clone().cpu()
    embed_dict[key].requires_grad = False  # Ensure no gradients are tracked

torch.save(embed_dict, f"dataset/ogbn_mag/processed/hpc/paper_dict_{embedding_dim}_{num_epochs}_epoch.pt")


print('Embed_batches done')