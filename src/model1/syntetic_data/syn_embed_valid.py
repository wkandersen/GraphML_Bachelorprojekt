import torch
import sys
import os
import gc
from Packages.mini_batches_fast import mini_batches_fast
from Packages.loss_function import LossFunction
# from Packages.synthetic_data_mixture import paper_c_paper_train, paper_c_paper_valid, data,venues_values
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator
import traceback
from collections import defaultdict
def set_seed(seed=45):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(45)

def plot_paper_venue_embeddings(
    venue_value, 
    embed_dict, 
    sample_size=100, 
    filename=None, 
    dim=2, 
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
    else:
        plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
embedding_dim = 2

save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b1.pt')
paper_c_paper_train,paper_c_paper_valid = save['paper_c_paper_train'], save['paper_c_paper_valid']
data,venue_value = save['data'], save['venue_value']
print(venue_value)

num_iterations = len(paper_c_paper_valid[0])


check = torch.load(f'src/model1/syntetic_data/embed_dict/embed_dict_test.pt', map_location=device)
paper_dict = check

# Move all embeddings to device
for ent_type in paper_dict:
    for k, v in paper_dict[ent_type].items():
        paper_dict[ent_type][k] = v.to(device)

citation_dict = defaultdict(list)
for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
    citation_dict[src.item()].append(tgt.item())

all_papers = list(citation_dict.keys())

loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

unique_train = set(paper_c_paper_train.flatten().unique().tolist())
unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())
valid_exclusive = unique_valid - unique_train
l_prev = list(valid_exclusive)
predictions = {}

for i in range(num_iterations):
    mini_b = mini_batches_fast(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'), data,citation_dict, all_papers,venues=False)
    dm, l_next, random_sample = mini_b.data_matrix()
    # print(random_sample)
    # print(paper_c_paper_valid)
    if len(dm) < 2:
        print(f"[SKIP] Too few nodes for venue comparison: {random_sample}")
        continue

    dm = dm[dm[:, 4] != 4]
    print(dm)

    test1 = dm[dm[:, 0] == 1]
    list_test = test1[:, 2].unique().tolist()

    # mini_btrain1 = mini_batches_fast(paper_c_paper_train, dm[:, 2].tolist(), len(dm[:, 2]), ('paper', 'cites', 'paper'), data,citation_dict, all_papers)
    # dmtrain1, ultrain1, random_sampletrain1 = mini_btrain1.data_matrix()

    # test1 = dmtrain1[dmtrain1[:, 4] != 4]
    # test2 = test1[test1[:, 0] == 1]
    # list_test = test2[:, 2].unique().tolist()

    # concat_dm = torch.cat((dm, dmtrain1), 0)
    # rows = [[0, random_sample[0], test_item, 0, 0] for test_item in list_test]
    # tensor_result = torch.tensor(rows).to(device)
    # concat_dm = concat_dm.to(device)
    # tensor_result = tensor_result.to(device)
    # final = torch.cat((concat_dm, tensor_result), 0).to(device)

    embedding_list = []
    for test_item in list_test:
        if test_item in paper_dict['paper']:
            embedding_list.append(paper_dict['paper'][test_item])

    if len(embedding_list) == 0:
        new_embedding = torch.nn.Parameter(torch.randn(2, device=device, requires_grad=True))
    else:
        mean_tensor = torch.mean(torch.stack(embedding_list), dim=0)
        new_embedding = torch.nn.Parameter(mean_tensor.clone().to(device))

        # print(f"Element-wise mean tensor: {mean_tensor}")



    paper_dict['paper'][random_sample[0]] = new_embedding
    # print(paper_dict['paper'][random_sample[0]])
    new_optimizer = torch.optim.Adam([paper_dict['paper'][random_sample[0]]], lr=lr)


    prev_losses = []
    patience = 5
    tolerance = 1e-4  # Define how close "very close" means

    for epoch in range(num_epochs):
        new_optimizer.zero_grad()
        loss = loss_function.compute_loss(paper_dict, dm)
        if loss is None:
            print(f"[SKIP] Loss computation failed for sample {random_sample[0]}.")
            break
        loss.backward()
        new_optimizer.step()
        
        current_loss = loss.detach().item()
        
        # Track loss history
        prev_losses.append(current_loss)

        if len(prev_losses) > patience:
            recent = prev_losses[-patience:]
            if max(recent) - min(recent) < tolerance:
                print(f"[EARLY STOP] Loss converged after {epoch + 1} epochs.")
                break

    # print(paper_dict['paper'][random_sample[0]])
    true_label = int(venue_value[random_sample[0]].cpu().numpy())

    logi_f = []
    
    for j in range(len(paper_dict['venue'])):
        paper_emb = paper_dict['paper'][random_sample[0]].to(device)
        venue_emb = paper_dict['venue'][j].to(device)
        dist = torch.norm(paper_emb - venue_emb) ** 2
        logi = torch.sigmoid(alpha - dist)
        logi_f.append((logi.item(), j))

    logits, node_ids = zip(*logi_f)
    logi_f_tensor = torch.tensor(logits, device=device)
    softma = F.softmax(logi_f_tensor, dim=0)

    # Rank true label
    sorted_probs, sorted_indices = torch.sort(softma, descending=True)
    ranked_venue_ids = [node_ids[i] for i in sorted_indices.tolist()]
    if true_label in ranked_venue_ids:
        true_class_rank = ranked_venue_ids.index(true_label) + 1  # 1-based
    else:
        true_class_rank = -1  # Not found (shouldn't happen)

    # Store prediction and rank info
    predicted_node_id = ranked_venue_ids[0]
    highest_prob_value = sorted_probs[0].item()
    predictions[random_sample[0]] = (true_label, predicted_node_id, true_class_rank)

    items = sorted(predictions.items())
    y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
    y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

    evaluator = Evaluator(name='ogbn-mag')
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

    l_prev = l_next
    new_embedding = new_embedding.detach()
    del new_optimizer

    if (i + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

save_dir = f'dataset/ogbn_mag/processed/Predictions'
os.makedirs(save_dir, exist_ok=True)
torch.save(predictions, os.path.join(save_dir, f'pred_dict_{embedding_dim}.pt'))

# Filter paper_dict to only include papers for which predictions were made
filtered_paper_dict = {
    'paper': {k: v for k, v in paper_dict['paper'].items() if k in predictions},
    'venue': paper_dict['venue']
}
num_papers = len(filtered_paper_dict['paper'])

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=filtered_paper_dict,sample_size=num_papers,plot_params={
        "alpha": alpha,
        "lambda": lam,
        "lr": lr,
        "epochs": num_epochs,
        "dim": embedding_dim
    })
