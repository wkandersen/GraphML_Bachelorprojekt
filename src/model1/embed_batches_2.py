import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from Packages.mini_batches import mini_batches_code
from Packages.mini_batches_fast import mini_batches_fast
from Packages.data_divide import paper_c_paper_train
from Packages.loss_function import LossFunction
from Packages.embed_trainer import NodeEmbeddingTrainer
import gc
import wandb
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

# Create a timestamp string for the run
run_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_dir = os.path.join("checkpoint", run_start_time)

# Create the directory
os.makedirs(checkpoint_dir, exist_ok=True)



def get_args():
    parser = argparse.ArgumentParser(description='Training configuration')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha hyperparameter')
    parser.add_argument('--lam', type=float, default=0.001, help='Lambda hyperparameter')
    parser.add_argument('--embedding_dim', type=int, default=2, help='Embedding Dimensions')
    parser.add_argument('--weight', type=float, default = 1.0, help = "Weight for non-edges")
    parser.add_argument('--iterations', type=bool, default=True, help = 'Number of iterations')
    parser.add_argument('--venue_weight', type=float, default = 1.0, help = "Weight for venue_edges")
    parser.add_argument('--neg_ratio', type=int, default = 5, help="Negative sample ratio")

    return parser.parse_args()


args = get_args()

batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr 
alpha = args.alpha
lam = args.lam
embedding_dim = args.embedding_dim
weight = args.weight
iterations = args.iterations
venue_weight = args.venue_weight
neg_ratio = args.neg_ratio

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")
loss_function = LossFunction(alpha=alpha, lam=lam, weight=weight, venue_weight=venue_weight, neg_ratio=neg_ratio)
N_emb = NodeEmbeddingTrainer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
checkpoint_epoch = 18

# Load initial embeddings
# embed_dict = torch.load(f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_1.pt", map_location=device)
checkpoint = torch.load(f'checkpoint/checkpoint_iter_64_{embedding_dim}_50_epoch_{checkpoint_epoch}_weight_0.1_with_optimizer.pt',map_location=device)
embed_dict = checkpoint['collected_embeddings']
optimizer_state = checkpoint['optimizer_state']
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device, weights_only=False)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

# optimizerstate = embed_dict['optimizer_state']
# embed_dict = embed_dict['collected_embeddings']


citation_dict = defaultdict(list)
for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
    citation_dict[src.item()].append(tgt.item())

all_papers = list(citation_dict.keys())


saved_checkpoints = []
max_saved = 2
save_every_iter = 1

if iterations == True:
    try:
        num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper'])) # we need to be able to look at the complete dataset
    except:
        num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper']))
else:
    num_iterations = 75

print(f'Batch size: {args.batch_size}')
print(f'Epochs: {args.epochs}')
print(f'Learning rate: {args.lr}')
print(f'Alpha: {args.alpha}')
print(f'Lambda: {args.lam}')
print(f'Embedding dim: {args.embedding_dim}')
print(f'Weight: {args.weight}')
print(f'Neg_ratio: {args.neg_ratio}')

run = wandb.init(
    project="Bachelor_projekt",
    name=f"part_4_5_neg_run_{datetime.now():%Y-%m-%d_%H-%M-%S}, {embedding_dim} and {venue_weight}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "alpha": alpha,
        "lam": lam,
        "weight": weight,
        "venue_weight": venue_weight,
        "neg_ratio": neg_ratio
    },
)


for entity_type, subdict in embed_dict.items():
    for key, tensor in subdict.items():
        param = torch.nn.Parameter(tensor.to(device), requires_grad=True)
        # Scale down from large initial range to prevent vanishing gradients
        param.data /= 100.0
        embed_dict[entity_type][key] = param

params = []
for group in embed_dict.values():  # e.g., embed_dict['paper'], embed_dict['venue']
    for param in group.values():   # e.g., embed_dict['paper'][123]
        params.append(param)


# 
optimizer = torch.optim.Adam(params, lr=lr)
optimizer.load_state_dict(optimizer_state)

loss_pr_epoch = []
for i in range(num_epochs):
    print(f"Epoch {i + 1}/{num_epochs}")
    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
    loss_pr_iteration = []

    # import time
    # start = time.time()
    # dm, unique_list, random_sample = mini_b.data_matrix()
    # print("Batch gen time:", time.time() - start)

    mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data, citation_dict, all_papers)

    for j in range(num_iterations):
        mini_b.set_unique_list(l_prev)  # Update only the node list
        dm, l_next, random_sample = mini_b.data_matrix()
        dm_ven_pap = dm[dm[:,4] == 4]
        dm_pap_pap = dm[dm[:,4] != 4]
        # print(dm)

    # for j in range(num_iterations):

        # Generate mini-batches
        # mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
        # dm, l_next, random_sample = mini_b.data_matrix()

        # Move data to GPU
        dm = dm.to(device)
        optimizer.zero_grad()
        loss = loss_function.compute_loss(embed_dict, dm)
        loss_ven = loss_function.compute_loss(embed_dict,dm_ven_pap)
        loss_pap = loss_function.compute_loss(embed_dict,dm_pap_pap)
        loss.backward()
        # print(embed_dict)
        optimizer.step()
        # Log loss to wandb
        wandb.log({"loss": loss.detach().item()})
        wandb.log({"paper_loss": loss_pap.detach().item()})
        wandb.log({"venue_loss": loss_ven.detach().item()})
        # print(f"Loss: {loss.detach().item()}")
        # Update node list for the next iteration
        loss_pr_iteration.append(loss.detach().item())

        l_prev = l_next
        

        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            # print(loss_pr_iteration)
            loss_pr_epoch.append(np.mean(loss_pr_iteration))
            wandb.log({"mean_loss_epoch": loss_pr_epoch[i]})
            print(loss_pr_epoch[i])
            break

        # # Cleanup
        # if (i + 1) % 25 == 0:  # Or do it every iteration if memory is super tight
        #     import gc
        #     gc.collect()
        #     torch.cuda.empty_cache()

    if (i + 1) % save_every_iter == 0:
        iter_id = i + 1

        # Define paths within timestamped folder
        trainer_path = os.path.join(checkpoint_dir, f"trainer_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        embed_path = os.path.join(checkpoint_dir, f"embed_dict_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        l_prev_path = os.path.join(checkpoint_dir, f"l_prev_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id+checkpoint_epoch}_weight_{weight}_with_optimizer.pt")


        # Save checkpoint with both embeddings and optimizer state
        checkpoint = {
            'collected_embeddings': {group_key: {id_key: tensor.cpu() for id_key, tensor in group.items()} for group_key, group in embed_dict.items()},
            'optimizer_state': optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)  # Save full checkpoint

        # Save trainer and embeddings separately
        # N_emb.save_checkpoint(trainer_path)
        # torch.save(l_prev, l_prev_path)

        # Append checkpoint paths to track for cleanup
        saved_checkpoints.append(checkpoint_path)

        # Remove older checkpoints if more than max_saved
        if len(saved_checkpoints) > max_saved:
            old_files = saved_checkpoints.pop(0)  # Get the oldest checkpoint
            if os.path.exists(old_files):
                os.remove(old_files)  # Delete the old checkpoint file
    
print(loss_pr_epoch)


for group_key in embed_dict:  # 'paper', 'venue'
    for id_key in embed_dict[group_key]:
        embed_dict[group_key][id_key] = embed_dict[group_key][id_key].detach().clone().cpu()

torch.save(embed_dict, f"dataset/ogbn_mag/processed/hpc/paper_dict_{batch_size}_{embedding_dim}_dim_{num_epochs+checkpoint_epoch+1}_epoch.pt")


print('Embed_batches done')