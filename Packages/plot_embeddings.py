import numpy as np
import matplotlib.pyplot as plt
import torch

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

venue_value =torch.load('dataset/ogbn_mag/processed/venue_value.pt')
saved = torch.load('checkpoint/checkpoint_iter_64_8_50_epoch_18_weight_0.1_with_optimizer.pt')
embed_dict = saved['collected_embeddings']
path = 'epoch_18_dim_8'

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

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=embed_dict,filename=path,dim=2)
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

