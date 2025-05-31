import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Packages.mini_batches import mini_batches_code
from Packages.data_divide import paper_c_paper_train
from Packages.loss_function import LossFunction
from Packages.embed_trainer import NodeEmbeddingTrainer
import gc
import wandb
from datetime import datetime
import argparse
import numpy as np


wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

loss_function = LossFunction()
N_emb = NodeEmbeddingTrainer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
embedding_dim = 2
# Load initial embeddings
embed_dict = torch.load(f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}.pt", map_location=device)
# hybrid_dict = torch.load(f"dataset/ogbn_mag/processed/hybrid_dict_{embedding_dim}_test.pt", map_location=device)
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device, weights_only=False)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

X = data.x_dict[('paper')].to(device)

hybrid_dict = {}

for key in embed_dict:
    hybrid_dict[key] = {}

    if key == 'venue':
        # Direct copy of venue embeddings
        for idx, embedding in embed_dict[key].items():
            hybrid_dict[key][idx] = embedding  # no clone, no concat
    else:
                # Concatenate embed with X and make it a leaf again
        for idx, embedding in embed_dict['paper'].items():
            hybrid_dict['paper'][idx] = torch.cat((embed_dict['paper'][idx], X[idx]), -1)


saved_checkpoints = []
max_saved = 2
save_every_iter = 5

def get_args():
    parser = argparse.ArgumentParser(description='Training configuration')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha hyperparameter')
    parser.add_argument('--lam', type=float, default=0.001, help='Lambda hyperparameter')

    return parser.parse_args()

args = get_args()
batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr
alpha = args.alpha
lam = args.lam


num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper'])) # we need to be able to look at the complete dataset

# num_iterations = 3

print(f'Batch size: {args.batch_size}')
print(f'Epochs: {args.epochs}')
print(f'Learning rate: {args.lr}')
print(f'Alpha: {args.alpha}')
print(f'Lambda: {args.lam}')

run = wandb.init(
    project="Bachelor_projekt",
    name=f"hybrid_run_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "alpha": alpha,
        "lam": lam
    },
)

params = []
for subdict in embed_dict.values():
    for k, v in subdict.items():
        if not v.requires_grad:
            v.requires_grad = True
        params.append(v)
loss_pr_epoch = []

optimizer = torch.optim.Adam(params, lr=lr)

for i in range(num_epochs):
    print(f"Epoch {i + 1}/{num_epochs}")
    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
    loss_pr_iteration = []

    for j in range(num_iterations):

        # Generate mini-batches
        mini_b = mini_batches_code(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
        dm, l_next, random_sample = mini_b.data_matrix()

        dm = dm[dm[:,4]!=4]
        combined_list = dm[:, 2].unique().tolist() + random_sample

        if j == 1 or j == 5:
            print(random_sample[0])
            print(embed_dict['paper'][random_sample[0]])
            print(hybrid_dict['paper'][random_sample[0]])
        
        # Move data to GPU
        dm = dm.to(device)
        optimizer.zero_grad()
        loss = loss_function.compute_loss(hybrid_dict, dm)
        loss.backward()
        optimizer.step()

        with torch.no_grad():  # Prevent this op from being tracked in autograd
            for idx in combined_list:
                hybrid_dict['paper'][idx] = torch.cat((
                    embed_dict['paper'][idx],  # This is being updated by optimizer
                    X[idx]
                ), dim=-1)



        # Log loss to wandb
        wandb.log({"loss": loss.detach().item()}, step=j + 1)
        print(f"Loss: {loss.detach().item()}")
        # Update node list for the next iteration
        loss_pr_iteration.append(loss.detach().item())

        l_prev = l_next

        if j == 1 or j == 5:
            print(embed_dict['paper'][random_sample[0]])
            print(hybrid_dict['paper'][random_sample[0]])
        

        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            print(loss_pr_iteration)
            loss_pr_epoch.append(np.mean(loss_pr_iteration))
            wandb.log({"loss_epoch": loss_pr_epoch[i]}, step=i+1)
            break

        # Cleanup
        if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    if (i + 1) % save_every_iter == 0:
        iter_id = i + 1

        os.makedirs("checkpoint_hybrid", exist_ok=True)
        embed_path = f"checkpoint_hybrid/embed_dict_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"
        hybrid_path = f"checkpoint_hybrid/hybrid_dict_iter_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt"

        torch.save(embed_dict,embed_path)
        torch.save(hybrid_dict,hybrid_path)

        saved_checkpoints.append((embed_path,hybrid_path))

        # Remove older checkpoints if more than max_saved
        if len(saved_checkpoints) > max_saved:
            old_files = saved_checkpoints.pop(0)  # Get the oldest checkpoint
            for f in old_files:
                if os.path.exists(f):
                    os.remove(f)  # Delete the old checkpoint file

print('Hybrid done')