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
# from Packages.create_syn_data import data,paper_c_paper_train, collected_embeddings,venue_value, embedding_dim,num_papers,b,venue_value_test,test_data
# from Packages.synthetic_data_mixture import data,paper_c_paper_train, collected_embeddings, embedding_dim,b
from venue_homogeneity import compute_venue_homogeneity
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

def plot_paper_venue_embeddings(
    venue_value, 
    embed_dict, 
    sample_size=num_papers, 
    filename=None, 
    dim=embedding_dim, 
    top=1,
    plot_params=None  # New parameter
):
    """
    Plots 2D projection of paper and venue embeddings, using PCA if original dim > 2.

    Parameters:
        venue_value (dict): Mapping from paper_id to true venue_id
        embed_dict (dict): Dictionary with 'paper' and 'venue' embedding tensors
        sample_size (int): Number of papers to sample
        filename (str or None): If provided, saves plot to this file
        dim (int): Original embedding dimension. If > 2, PCA is applied to reduce to 2D.
        top (int): How many closest venues to consider as correct match.
        plot_params (dict or None): Dictionary of parameters to annotate in the plot.
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
    count_true_in_topk = 0

    for idx, paper_id in enumerate(sampled_paper_keys):
        paper_vec = paper_points[idx]
        true_venue = true_labels.get(paper_id, None)

        if true_venue is None or true_venue not in venue_id_to_index:
            paper_colors.append(-1)
            continue

        dists = np.linalg.norm(venue_points - paper_vec, axis=1)
        nearest_indices = np.argsort(dists)[:top]
        nearest_venues = [venue_ids[i] for i in nearest_indices]

        if true_venue in nearest_venues:
            paper_colors.append(venue_to_color[true_venue])
            count_true_in_topk += 1
        else:
            paper_colors.append(-1)

    paper_colors = np.array(paper_colors)

    # Plotting
    plt.figure(figsize=(12, 8))
    mask = paper_colors != -1

    scatter = plt.scatter(
        paper_points[mask, 0], paper_points[mask, 1],
        c=paper_colors[mask], s=2, alpha=0.6,
        cmap='tab20', label=f'Papers (true venue in top {top})'
    )

    if np.any(paper_colors == -1):
        plt.scatter(
            paper_points[paper_colors == -1, 0],
            paper_points[paper_colors == -1, 1],
            s=2, alpha=0.3, color='gray',
            label=f'Papers (true venue NOT in top {top})'
        )

    plt.scatter(
        venue_points[:, 0], venue_points[:, 1],
        s=20, alpha=0.9, color='red', label='Venues'
    )

    plt.legend()
    title = f'Sampled Paper and Venue Embeddings\nTrue venue in top {top}: {count_true_in_topk} / {len(paper_points)}'
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Colorbar
    cbar = plt.colorbar(scatter, ticks=range(len(unique_venues)))
    cbar.ax.set_yticklabels([str(v) for v in unique_venues])
    cbar.set_label('Venue ID')

    # Optional parameters text block
    if plot_params:
        param_text = "\n".join([f"{k}={v}" for k, v in plot_params.items()])
        plt.gcf().text(0.01, 0.01, param_text, fontsize=8, va='bottom', ha='left')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to: {filename}")
        plt.close()
    # else:
    #     plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_pos_neg_histograms(pos_probs, neg_probs, epoch, batch):
    plt.figure(figsize=(10, 5))

    bins = np.linspace(0, 1, 50)

    # Plot positive edge probs as blue step outline
    plt.hist(pos_probs, bins=bins, density=True, histtype='step', linewidth=2,
             color='blue', label='Positive edges (y=1)')

    # Plot negative edge probs as red step outline
    plt.hist(neg_probs, bins=bins, density=True, histtype='step', linewidth=2,
             color='red', label='Negative edges (y=0)')

    # Optional: plot means as vertical lines
    plt.axvline(pos_probs.mean(), color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(neg_probs.mean(), color='red', linestyle='--', linewidth=1.5)

    plt.xlabel('Predicted probability p(y=1|z_u,z_v)')
    plt.ylabel('Density')
    plt.title(f'Predicted Probability Distribution\nEpoch {epoch}, Batch {batch}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

embed_dict = save['collected_embeddings']

batch_size = 60
num_epochs = 25
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
        loss_ven = loss_function.compute_loss(embed_dict,dm_ven_pap)
        loss_pap = loss_function.compute_loss(embed_dict,dm_pap_pap)
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
        if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
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

