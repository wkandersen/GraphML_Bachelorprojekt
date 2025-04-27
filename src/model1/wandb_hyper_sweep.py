import torch
import copy
import sys
import os
import wandb
import random
import numpy as np

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sweep configuration
sweep_configuration = {
    "method": "random",  # Can also use "bayes"
    "name": "embedding_hyperparam_sweep",
    "metric": {"goal": "minimize", "name": "final_loss"},
    "parameters": {
        "batch_size": {"values": [100,500,700]},  # Must remain as list to avoid type issues
        "num_epochs": {"values": [10,30,50]},
        "lr": {"values": [0.01, 0.1]},
        "alpha": {"values": [0.5,0.1, 0.05]},
        "lam": {"values": [0.001, 0.01,0.1]}
    }
}

# Sweep function
def sweep_objective():
        # Set a fixed seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run = wandb.init()
    config = wandb.config

    # Load initial embeddings
    embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt", map_location=device)
    embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt", map_location=device)
    data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

    # Clone dictionaries
    paper_dict = {k: v.clone() for k, v in embed_paper.items()}
    venue_dict = {k: v.clone() for k, v in embed_venue.items()}

    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial node list

    num_iterations = 1500  # Adjust as needed
    for i in range(num_iterations):
        print(f"Iteration {i+1}")
        print(f"Config: {dict(config)}")

        # Generate mini-batch
        mini_b = mini_batches_code(paper_c_paper_train, l_prev, config.batch_size, ('paper', 'cites', 'paper'),data)
        dm, l_next, remapped_datamatrix_tensor, _ = mini_b.node_mapping()

        # Move data to device
        dm = dm.to(device)
        remapped_datamatrix_tensor = remapped_datamatrix_tensor.to(device)

        # Train
        N_emb = NodeEmbeddingTrainer(
            dm=dm,
            remapped_datamatrix_tensor=remapped_datamatrix_tensor,
            paper_dict=paper_dict,
            venue_dict=venue_dict,
            num_epochs=config.num_epochs,
            lr=config.lr,
            alpha=config.alpha,
            lam=config.lam,
            device=device
        )

        paper_dict, venue_dict, loss = N_emb.train()
        l_prev = l_next

        wandb.log({"final_loss": loss})

    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Bachelor_projekt")
    wandb.agent(sweep_id, function=sweep_objective, count=15)  # Run 15 trials
