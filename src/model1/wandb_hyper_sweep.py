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
from Packages.data_divide import paper_c_paper_train
from Packages.loss_function import LossFunction

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sweep configuration
sweep_configuration = {
    "method": "random",  # Can also use "bayes"
    "name": "test1",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "lam": {"values": [0,0.001, 0.01,0.1]},
        "alpha": {"values": [0.01,0.1,0.5,1]},
        "lr": {"values": [0.1,0.01]},
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

    loss_function = LossFunction(alpha=config.alpha,lam=config.lam)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("starting")
    embedding_dim = 8
    # Load initial embeddings
    embed_dict = torch.load(f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}.pt", map_location=device)
    data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

    batch_size = 200
    num_epochs = 1
    num_iterations = 750 #forhåbentlig 25% af datasættet

    print(f'Batch_size: {batch_size}')
    print(f'Num_epochs: {num_epochs}')
    print(f'Num_iterations: {num_iterations}')
    print(f'Embed_dim: {embed_dict}')

    # batch_size = 10
    # num_epochs = 1
    # num_iterations = 5 #forhåbentlig 25% af datasættet

    params = []
    for subdict in embed_dict.values():
        params.extend(subdict.values())

    for i in range(num_epochs):
        print(f"Epoch {i + 1}/{num_epochs}")
        l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
        optimizer = torch.optim.Adam(params, lr=config.lr)
        loss_pr_iteration = []


        for j in range(num_iterations):

            # Generate mini-batches
            mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
            dm, l_next, random_sample = mini_b.data_matrix()

            # Move data to GPU
            dm = dm.to(device)
            optimizer.zero_grad()
            loss = loss_function.compute_loss(embed_dict, dm)
            loss.backward()
            optimizer.step()
            # Log loss to wandb
            wandb.log({"loss": loss.detach().item()})
            print(f"Loss: {loss.detach().item()}")
            # Update node list for the next iteration
            loss_pr_iteration.append(loss.detach().item())

            l_prev = l_next

    
    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Bachelor_projekt")
    wandb.agent(sweep_id, function=sweep_objective, count=16)  # Run 15 trials
