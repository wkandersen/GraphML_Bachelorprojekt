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
from venue_homogeneity import compute_venue_homogeneity
from pprint import pprint
from torch.nn import Parameter
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance


embedding_dim = 2
b = 1
save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b{b}.pt')
paper_c_paper_train = save['paper_c_paper_train']
data,venue_value = save['data'], save['venue_value']
num_papers = len(paper_c_paper_train.unique())
set_seed(69)
text = 'test'

overall, per_paper = compute_venue_homogeneity(paper_c_paper_train, venue_value)
print(f"Overall venue homogeneity: {overall:.4f}")

embed_dict = save['collected_embeddings']

batch_size = 60
num_epochs = 2
lr = 0.5
alpha = 0.1
lam = 0.1
weight = 1
venue_weight = 100
neg_ratio = 5

folder = 'test'
# folder_specifik = f'false'
folder_specifik = f''
# file_before = 'before_200_base'
file_after = f'src/model1/syntetic_data/Plots/{folder}/after_{folder_specifik}.png'

loss_function = LossFunction(alpha=alpha,weight=weight,lam=lam,venue_weight=venue_weight,neg_ratio=neg_ratio)
N_emb = NodeEmbeddingTrainer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting") 


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
        loss = loss_function.compute_loss(embed_dict, dm)
        if len(dm_pap_pap) > 0:
            loss_pap = loss_function.compute_loss(embed_dict,dm_pap_pap)
        loss_ven = loss_function.compute_loss(embed_dict,dm_ven_pap)
        loss.backward()
        optimizer.step()

                # Get predicted probabilities and labels for this batch
        probs, labels = loss_function.get_probs_and_labels(embed_dict, dm)

        # Move to CPU and numpy for numpy histogram plotting
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_pos_probs.append(probs_np[labels_np == 1])
        all_neg_probs.append(probs_np[labels_np == 0])

        # print(f"Loss: {loss.detach().item()}")
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
            print(f"loss_epoch: {loss_pr_epoch[i]}")
            break

        # Cleanup
        if (i + 1) % 50 == 0:  # Or do it every iteration if memory is super tight
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    if i % 50 == 0:
        pos_probs_epoch = np.concatenate(all_pos_probs)
        neg_probs_epoch = np.concatenate(all_neg_probs)

        plot_pos_neg_histograms(pos_probs_epoch, neg_probs_epoch, epoch=i, batch='all')

# # After training
with open("embedding_output_after.txt", "w") as f:
    pprint(embed_dict, stream=f, indent=2, width=80)

import os
print("Saving to:", os.getcwd())

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=embed_dict,sample_size=num_papers,filename=file_after,plot_params={
        "alpha": alpha,
        "lambda": lam,
        "weight": weight,
        "venue_weight": venue_weight,
        "lr": lr,
        "epochs": num_epochs,
        "dim": embedding_dim
    })

# Plot training loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_pr_epoch) + 1), loss_pr_epoch, marker='o', linestyle='-')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# Add parameters to loss plot
params_text = (
    f"num_papers={num_papers}, b={b}, weight={weight}, venue_weight={venue_weight},\n"
    f"alpha={alpha}, lam={lam}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}"
)
plt.annotate(params_text, xy=(0.5, 0.95), xycoords='axes fraction',
             fontsize=9, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

plt.tight_layout()

# Optional: Save to file
loss_plot_path = f"src/model1/syntetic_data/Plots/{folder}/loss_plot_{folder_specifik}.png"
plt.savefig(loss_plot_path, dpi=300)
print(f"Loss plot saved to: {loss_plot_path}")
# plt.show()


# with open("embedding_output_before.txt") as f1, open("embedding_output_after.txt") as f2:
#     for i, (line1, line2) in enumerate(zip(f1, f2), 1):
#         if line1 != line2:
#             print(f"Line {i} does not differs:")
#             print(f"  before: {line1.strip()}")
#             print(f"  after : {line2.strip()}")

plt.figure(figsize=(8, 5))

# Plot both losses
plt.plot(range(1, len(loss_ven_epoch) + 1), loss_ven_epoch, marker='o', linestyle='-', label='Venue Loss')
plt.plot(range(1, len(loss_pap_epoch) + 1), loss_pap_epoch, marker='s', linestyle='--', label='Paper Loss')

plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Add legend to distinguish the lines
plt.legend()

# Annotate with parameters
params_text = (
    f"num_papers={num_papers}, b={b}, weight={weight}, venue_weight={venue_weight},\n"
    f"alpha={alpha}, lam={lam}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}"
)
plt.annotate(params_text, xy=(0.5, 0.95), xycoords='axes fraction',
             fontsize=9, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

plt.tight_layout()

# Save to file
loss_plot_path = f"src/model1/syntetic_data/Plots/{folder}/loss_ven_pap_plot_{folder_specifik}.png"
plt.savefig(loss_plot_path, dpi=300)
print(f"Loss plot saved to: {loss_plot_path}")
# plt.show()

for group_key in embed_dict:  # 'paper', 'venue'
    for id_key in embed_dict[group_key]:
        embed_dict[group_key][id_key] = embed_dict[group_key][id_key].detach().clone().cpu()

torch.save(embed_dict, f"src/model1/syntetic_data/embed_dict/embed_dict_{text}.pt")