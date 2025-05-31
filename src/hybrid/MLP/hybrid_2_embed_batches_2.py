import torch
import copy
import sys
import os
import gc
import wandb
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from Packages.loss_function import LossFunction
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.mini_batches_fast import mini_batches_fast
from Packages.plot_embeddings import set_seed, plot_paper_venue_embeddings, plot_pos_neg_histograms
from Packages.data_divide import paper_c_paper_train, data, venue_value
# from src.venue_homogeneity import compute_venue_homogeneity
from pprint import pprint
from torch.nn import Parameter
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

run_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_dir = os.path.join("checkpoint", run_start_time)

# Create the directory
os.makedirs(checkpoint_dir, exist_ok=True)

embedding_dim = 2
b = 1
saved_checkpoints = []
max_saved = 2
save_every_iter = 1
save = torch.load(f'dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_{b}_hybrid.pt', map_location=device, weights_only=False)
embed_dict = save
# paper_c_paper_train = save['paper_c_paper_train']
# data,venue_value = save['data'], save['venue_value']
# embed_dict = save['collected_embeddings']
# emb2d = save['emb2d']
emb2d  = torch.load('src/hybrid/MLP/embeddings/train_embeddings_dict_max_epochs1.pt', map_location=device, weights_only=False)
num_papers = len(paper_c_paper_train.unique())
set_seed(69)
text = 'test'



# overall, per_paper = compute_venue_homogeneity(paper_c_paper_train, venue_value)
# print(f"Overall venue homogeneity: {overall:.4f}")


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

folder = 'test'
# folder_specifik = f'false'
folder_specifik = f''
# file_before = 'before_200_base'
file_after = f'src/hybrid/plots/{folder}/after_{folder_specifik}.png'

loss_function = LossFunction(alpha=alpha,weight=weight,lam=lam,venue_weight=venue_weight,neg_ratio=neg_ratio)
N_emb = NodeEmbeddingTrainer()


print("starting") 

run = wandb.init(
    project="Bachelor_projekt",
    name=f"hybrid_embed_batches_{datetime.now():%Y-%m-%d_%H-%M-%S}, {embedding_dim} and {venue_weight}",
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

citation_dict = defaultdict(list)
for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
    citation_dict[src.item()].append(tgt.item())

all_papers = list(citation_dict.keys())

num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper'])) # we need to be able to look at the complete dataset

# num_iterations = 2

# Before training
with open("embedding_output_before.txt", "w") as f:
    pprint(embed_dict, stream=f, indent=2, width=80)

# plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=embed_dict,sample_size=num_papers,filename=file_before)

for group in embed_dict:
    for key in embed_dict[group]: 
        embed_dict[group][key] = Parameter(
            embed_dict[group][key].clone().detach().to(device),
            requires_grad=True
        )

params = []
for group in embed_dict.values():  # e.g., embed_dict['paper'], embed_dict['venue']
    for param in group.values():   # e.g., embed_dict['paper'][123]
        params.append(param)

optimizer = torch.optim.Adam(params, lr=lr)

loss_pr_epoch = []
loss_ven_epoch = []
loss_pap_epoch = []
for i in range(num_epochs):
    print(f"Epoch {i + 1}/{num_epochs}")
    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
    loss_pr_iteration = []
    loss_ven_iteration = []
    loss_pap_iteration = []
    all_pos_probs = []
    all_neg_probs = []

    mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data, citation_dict, all_papers)

    for j in range(num_iterations):
        mini_b.set_unique_list(l_prev)  # Update only the node list
        dm, l_next, random_sample = mini_b.data_matrix()
        dm_ven_pap = dm[dm[:,4] == 4]
        dm_pap_pap = dm[dm[:,4] != 4]
        # print(random_sample)
        # print(dm)

    # for j in range(num_iterations):

        # Generate mini-batches
        # mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
        # dm, l_next, random_sample = mini_b.data_matrix()

        # Move data to GPU
        dm = dm.to(device)
        optimizer.zero_grad()
        loss = loss_function.compute_loss_hybrid(embed_dict, dm, emb2d)
        if len(dm_pap_pap) > 0:
            loss_pap = loss_function.compute_loss_hybrid(embed_dict, dm_pap_pap,emb2d)
        loss_ven = loss_function.compute_loss_hybrid(embed_dict, dm_ven_pap,emb2d)
        # print(f"before: {embed_dict['paper'][random_sample[0]]}")
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.detach().item()})
        wandb.log({"paper_loss": loss_pap.detach().item()})
        wandb.log({"venue_loss": loss_ven.detach().item()})
        # print(f"after: {embed_dict['paper'][random_sample[0]]}")

                # Get predicted probabilities and labels for this batch
        probs, labels = loss_function.get_probs_and_labels_hybrid(embed_dict, dm,emb2d)

        # Move to CPU and numpy for numpy histogram plotting
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_pos_probs.append(probs_np[labels_np == 1])
        all_neg_probs.append(probs_np[labels_np == 0])

        print(f"Loss: {loss.detach().item()}")
        # Update node list for the next iteration
        loss_pr_iteration.append(loss.detach().item())
        loss_ven_iteration.append(loss_ven.detach().item())
        if len(dm_pap_pap) > 0:
            loss_pap_iteration.append(loss_pap.detach().item())

        l_prev = l_next


        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            print(loss_pr_iteration)
            loss_pr_epoch.append(np.mean(loss_pr_iteration))
            loss_ven_epoch.append(np.mean(loss_ven_iteration))
            loss_pap_epoch.append(np.mean(loss_pap_iteration))
            wandb.log({"loss_epoch": loss_pr_epoch[i]})
            wandb.log({"mean_venue_loss_epoch": loss_ven_epoch[i]})
            wandb.log({"mean_paper_loss_epoch": loss_pap_epoch[i]})
            print(f"loss_epoch: {loss_pr_epoch[i]}")
            break

        # # Cleanup
        # if (i + 1) % 50 == 0:  # Or do it every iteration if memory is super tight
        #     import gc
        #     gc.collect()
        #     torch.cuda.empty_cache()

    if (i + 1) % save_every_iter == 0:
        iter_id = i + 1

        # Define paths within timestamped folder
        trainer_path = os.path.join(checkpoint_dir, f"trainer_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        embed_path = os.path.join(checkpoint_dir, f"embed_dict_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        l_prev_path = os.path.join(checkpoint_dir, f"l_prev_hybrid_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id}.pt")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_hybrid_iter_{batch_size}_{embedding_dim}_{num_epochs}_epoch_{iter_id+checkpoint_epoch}_weight_{weight}_with_optimizer.pt")

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

torch.save(embed_dict, f"dataset/ogbn_mag/processed/hpc/paper_dict_{batch_size}_{embedding_dim}_dim_{num_epochs}_epoch.pt")


print('Embed_batches done')
#     if i % 50 == 0:
#         pos_probs_epoch = np.concatenate(all_pos_probs)
#         neg_probs_epoch = np.concatenate(all_neg_probs)

#         plot_pos_neg_histograms(pos_probs_epoch, neg_probs_epoch, epoch=i, batch='all')

# # # After training
# with open("embedding_output_after.txt", "w") as f:
#     pprint(embed_dict, stream=f, indent=2, width=80)

# import os
# print("Saving to:", os.getcwd())

# full_dict = copy.deepcopy(embed_dict)
# #concatenate full dict['paper'] with emb2d
# for key in full_dict['paper']:
#     full_dict['paper'][key] = torch.cat((full_dict['paper'][key], emb2d[key]), dim=0)

# plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=full_dict,sample_size=num_papers,filename=file_after,plot_params={
#         "alpha": alpha,
#         "lambda": lam,
#         "weight": weight,
#         "venue_weight": venue_weight,
#         "lr": lr,
#         "epochs": num_epochs,
#         "dim": embedding_dim
#     })

# # Plot training loss over epochs
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(loss_pr_epoch) + 1), loss_pr_epoch, marker='o', linestyle='-')
# plt.title('Training Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# # Add parameters to loss plot
# params_text = (
#     f"num_papers={num_papers}, b={b}, weight={weight}, venue_weight={venue_weight},\n"
#     f"alpha={alpha}, lam={lam}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}"
# )
# plt.annotate(params_text, xy=(0.5, 0.95), xycoords='axes fraction',
#              fontsize=9, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

# plt.tight_layout()

# # Optional: Save to file
# loss_plot_path = f"src/hybrid/plots/test/loss_plot{folder_specifik}.png"
# plt.savefig(loss_plot_path, dpi=300)
# print(f"Loss plot saved to: {loss_plot_path}")
# # plt.show()


# # with open("embedding_output_before.txt") as f1, open("embedding_output_after.txt") as f2:
# #     for i, (line1, line2) in enumerate(zip(f1, f2), 1):
# #         if line1 != line2:
# #             print(f"Line {i} does not differs:")
# #             print(f"  before: {line1.strip()}")
# #             print(f"  after : {line2.strip()}")

# plt.figure(figsize=(8, 5))

# # Plot both losses
# plt.plot(range(1, len(loss_ven_epoch) + 1), loss_ven_epoch, marker='o', linestyle='-', label='Venue Loss')
# plt.plot(range(1, len(loss_pap_epoch) + 1), loss_pap_epoch, marker='s', linestyle='--', label='Paper Loss')

# plt.title('Training Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)

# # Add legend to distinguish the lines
# plt.legend()

# # Annotate with parameters
# params_text = (
#     f"num_papers={num_papers}, b={b}, weight={weight}, venue_weight={venue_weight},\n"
#     f"alpha={alpha}, lam={lam}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}"
# )
# plt.annotate(params_text, xy=(0.5, 0.95), xycoords='axes fraction',
#              fontsize=9, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

# plt.tight_layout()

# # Save to file
# loss_plot_path = f"src/hybrid/plots/{folder}/loss_ven_pap_plot_{folder_specifik}.png"
# plt.savefig(loss_plot_path, dpi=300)
# print(f"Loss plot saved to: {loss_plot_path}")
# # plt.show()

# for group_key in embed_dict:  # 'paper', 'venue'
#     for id_key in embed_dict[group_key]:
#         embed_dict[group_key][id_key] = embed_dict[group_key][id_key].detach().clone().cpu()

# torch.save(embed_dict, f"src/hybrid/MLP/embeddings/embed_dict_{text}.pt")