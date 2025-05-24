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
from src.mini_batches_fast import mini_batches_fast
from Packages.create_syn_data import data,paper_c_paper_train, collected_embeddings,venue_value, embedding_dim,num_papers
from pprint import pprint
from torch.nn import Parameter
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def plot_paper_venue_embeddings(venue_value, embed_dict, sample_size=num_papers, filename=None, dim=embedding_dim):
    """
    Plots 2D projection of paper and venue embeddings, using PCA if original dim > 2.

    Parameters:
        venue_value (dict): Mapping from paper_id to true venue_id
        embed_dict (dict): Dictionary with 'paper' and 'venue' embedding tensors
        sample_size (int): Number of papers to sample
        filename (str or None): If provided, saves plot to this file
        dim (int): Original embedding dimension. If > 2, PCA is applied to reduce to 2D.
    """
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    paper_embeddings = embed_dict['paper']
    venue_embeddings = embed_dict.get('venue', {})

    true_labels = {int(k): int(v) for k, v in venue_value.items()}

    paper_keys = list(paper_embeddings.keys())
    sampled_paper_keys = random.sample(paper_keys, sample_size) if len(paper_keys) > sample_size else paper_keys

    venue_ids = list(venue_embeddings.keys())
    if not venue_ids:
        raise ValueError("No venue embeddings found.")

    # Extract vectors
    paper_points_raw = np.array([paper_embeddings[k].detach().cpu().numpy() for k in sampled_paper_keys])
    venue_points_raw = np.array([venue_embeddings[k].detach().cpu().numpy() for k in venue_ids])

    # Combine for PCA if needed
    if dim > 2:
        combined = np.vstack([paper_points_raw, venue_points_raw])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        paper_points = combined_2d[:len(paper_points_raw)]
        venue_points = combined_2d[len(paper_points_raw):]
    else:
        paper_points = paper_points_raw
        venue_points = venue_points_raw

    # Map venue ID to index
    venue_id_to_index = {vid: idx for idx, vid in enumerate(venue_ids)}
    unique_venues = sorted(set(true_labels.values()))
    venue_to_color = {venue_id: idx for idx, venue_id in enumerate(unique_venues)}

    paper_colors = []
    count_true_in_top5 = 0

    for idx, paper_id in enumerate(sampled_paper_keys):
        paper_vec = paper_points[idx]
        true_venue = true_labels.get(paper_id, None)

        if true_venue is None or true_venue not in venue_id_to_index:
            paper_colors.append(-1)
            continue

        dists = np.linalg.norm(venue_points - paper_vec, axis=1)
        nearest_indices = np.argsort(dists)[:1]
        nearest_venues = [venue_ids[i] for i in nearest_indices]


        if true_venue in nearest_venues:
            paper_colors.append(venue_to_color[true_venue])
            count_true_in_top5 += 1
        else:
            paper_colors.append(-1)

    paper_colors = np.array(paper_colors)

    # Plotting
    plt.figure(figsize=(12, 8))
    mask = paper_colors != -1

    scatter = plt.scatter(
        paper_points[mask, 0], paper_points[mask, 1],
        c=paper_colors[mask], s=2, alpha=0.6,
        cmap='tab20', label='Papers (true venue in top 1)'
    )

    if np.any(paper_colors == -1):
        plt.scatter(
            paper_points[paper_colors == -1, 0],
            paper_points[paper_colors == -1, 1],
            s=2, alpha=0.3, color='gray',
            label='Papers (true venue NOT in top 5)'
        )

    plt.scatter(
        venue_points[:, 0], venue_points[:, 1],
        s=20, alpha=0.9, color='red', label='Venues'
    )

    plt.legend()
    plt.title(f'Sampled Paper and Venue Embeddings\nTrue venue in top 1: {count_true_in_top5} / {len(paper_points)}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Colorbar
    cbar = plt.colorbar(scatter, ticks=range(len(unique_venues)))
    cbar.ax.set_yticklabels([str(v) for v in unique_venues])
    cbar.set_label('Venue ID')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

embed_dict = collected_embeddings

batch_size = 35
num_epochs = 50
lr = 0.1
alpha = 0.1
lam = 0.01
weight = 0.001
venue_weight = 30
# file_before = 'before_200_base'
file_after = 'after_200_30_1'

loss_function = LossFunction(alpha=alpha,weight=weight,lam=lam,venue_weight=venue_weight)
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
for i in range(num_epochs):
    print(f"Epoch {i + 1}/{num_epochs}")
    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
    loss_pr_iteration = []

    mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data, citation_dict, all_papers)

    for j in range(num_iterations):
        mini_b.set_unique_list(l_prev)  # Update only the node list
        dm, l_next, random_sample = mini_b.data_matrix()
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
        loss.backward()
        optimizer.step()
        # print(f"Loss: {loss.detach().item()}")
        # Update node list for the next iteration
        loss_pr_iteration.append(loss.detach().item())

        l_prev = l_next
        

        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            print(loss_pr_iteration)
            loss_pr_epoch.append(np.mean(loss_pr_iteration))
            print(f"loss_epoch: {loss_pr_epoch[i]}")
            break

        # Cleanup
        if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
            import gc
            gc.collect()
            torch.cuda.empty_cache()

# After training
with open("embedding_output_after.txt", "w") as f:
    pprint(embed_dict, stream=f, indent=2, width=80)

import os
print("Saving to:", os.getcwd())

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=embed_dict,sample_size=num_papers,filename=file_after)

# with open("embedding_output_before.txt") as f1, open("embedding_output_after.txt") as f2:
#     for i, (line1, line2) in enumerate(zip(f1, f2), 1):
#         if line1 != line2:
#             print(f"Line {i} does not differs:")
#             print(f"  before: {line1.strip()}")
#             print(f"  after : {line2.strip()}")