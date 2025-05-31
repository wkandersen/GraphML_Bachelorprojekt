import torch
import sys
import os
import gc
from Packages.mini_batches_fast import mini_batches_fast
from Packages.loss_function import LossFunction
from Packages.plot_embeddings import set_seed, plot_paper_venue_embeddings
from Packages.data_divide import paper_c_paper_train, paper_c_paper_test, data
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb

def get_mean_median_embedding(paper_dict, ent_type='paper', method='mean'):
    # Extract all embeddings for the given entity type
    embeddings = list(paper_dict[ent_type].values())  # list of tensors

    if not embeddings:
        raise ValueError(f"No embeddings found for entity type: {ent_type}")

    # Stack into a single tensor of shape [N, D]
    stacked = torch.stack(embeddings, dim=0)

    if method == 'mean':
        return stacked.mean(dim=0)  # [D]
    elif method == 'median':
        return stacked.median(dim=0).values  # [D]
    else:
        raise ValueError(f"Unsupported method: {method}")


set_seed(45)
# wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 8        # Example: process 8 test nodes at once
embedding_dim = 2

# run = wandb.init(
#     project="Bachelor_projekt",
#     name=f"test_run_{datetime.now():%Y-%m-%d_%H-%M-%S}",
#     config={
#         "batch_size": batch_size,
#         "num_epochs": num_epochs,
#         "lr": lr,
#         "alpha": alpha,
#         "lam": lam,
#         "emb_dim": embedding_dim
#     },
# )

# Load checkpointed embeddings and venue mappings

save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b1.pt')
paper_c_paper_train,paper_c_paper_test = save['paper_c_paper_train'], save['paper_c_paper_test']
data,venue_value = save['data'], save['venue_value']
paper_dict = save['collected_embeddings']

# check = torch.load(
#     'checkpoint/2025-05-29_17-41-08/checkpoint_iter_64_2_50_epoch_58_weight_0.1_with_optimizer.pt',
#     map_location=device
# )
# paper_dict = check['collected_embeddings']
# venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device)

# Transfer all existing embeddings to device
for ent_type in paper_dict:
    for k, v in paper_dict[ent_type].items():
        paper_dict[ent_type][k] = v.to(device)

# Precompute mean and median embeddings over all papers
mean_emb = get_mean_median_embedding(paper_dict=paper_dict, method='mean')
median_emb = get_mean_median_embedding(paper_dict=paper_dict, method='median')

# Build citation dictionary for quick neighbor lookups
citation_dict = defaultdict(list)
for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
    citation_dict[int(src.item())].append(int(tgt.item()))

all_papers = list(citation_dict.keys())

loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

# Build sets of unique train/test papers
unique_train_set = set(paper_c_paper_train.flatten().unique().tolist())
unique_test_set  = set(paper_c_paper_test.flatten().unique().tolist())

# Among test set, we only want those not seen in train
valid_exclusive = unique_test_set - unique_train_set
l_prev = list(valid_exclusive)       # remaining test nodes to process (as Python list)

predictions = {}
acc = []

# Number of batches to process: ceil(|valid_exclusive| / batch_size)
num_iterations = (len(valid_exclusive) + batch_size - 1) // batch_size

for i in range(num_iterations):
    # Build a tensor from the current list of remaining test nodes
    unique_tensor = torch.tensor(l_prev, dtype=torch.long, device=device)

    mini_b = mini_batches_fast(
        paper_c_paper_test,      # full test edge list
        unique_tensor,           # remaining test node IDs
        batch_size,              # how many test nodes to sample at once
        ('paper', 'cites', 'paper'),
        data,
        citation_dict,
        all_papers,
        venues=True
    )

    dm, l_next, random_sample = mini_b.data_matrix()

    # If no more nodes to sample, break
    if len(random_sample) == 0:
        break

    # Convert random_sample (python list) to a tensor of node IDs
    batch_nodes = torch.tensor(random_sample, dtype=torch.long, device=device)

    # Filter out any “venue == 4” rows from the data matrix
    #    (these are placeholder rows if needed; specific to your data format)
    dm = dm[dm[:, 4] != 4]

    # Prepare to collect new embeddings for each node in batch_nodes
    new_params = []
    for test_node in random_sample:
        # Initialize new embedding: either mean or mean of neighbors
        neighbors = []
        # Collect neighbors where there is a “paper -> test_node” edge
        # In dm, column 0 = indicator (1 for real edge), col 2 = target ID
        if dm.shape[0] > 0:
            # mask rows where dm[:, 0] == 1 AND dm[:, 2] == test_node
            mask_pos = (dm[:, 0] == 1) & (dm[:, 2] == test_node)
            neighbor_ids = dm[mask_pos][:, 1].tolist()  # source IDs
            for nid in neighbor_ids:
                if int(nid) in paper_dict['paper']:
                    neighbors.append(paper_dict['paper'][int(nid)])

        if len(neighbors) == 0:
            # If no neighbors, initialize at global mean
            new_emb = torch.nn.Parameter(mean_emb.clone().to(device))
        else:
            # Otherwise, initialize to mean of neighbor embeddings
            stacked = torch.stack(neighbors, dim=0)
            mean_tensor = stacked.mean(dim=0)
            new_emb = torch.nn.Parameter(mean_tensor.clone().to(device))

        # Store in your dictionary so that loss_function can see it
        paper_dict['paper'][test_node] = new_emb
        new_params.append(new_emb)

    # Now we have a fresh embedding Parameter for each of the batch nodes.
    # Create a single optimizer over all of them:
    batch_optimizer = torch.optim.Adam(new_params, lr=lr)

    # Perform up to num_epochs of joint optimization over all batch nodes
    prev_losses = []
    patience = 5
    tolerance = 1e-4

    for epoch in range(num_epochs):
        batch_optimizer.zero_grad()
        loss = loss_function.compute_loss(paper_dict, dm)
        if loss is None:
            # In case compute_loss fails for this batch, skip further epochs
            break

        loss.backward()
        batch_optimizer.step()

        current_loss = loss.item()
        prev_losses.append(current_loss)
        # wandb.log({"batch_loss": current_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {current_loss:.4f}")

        # Early stopping if the last `patience` losses differ by < tolerance
        if len(prev_losses) > patience:
            recent = prev_losses[-patience:]
            if max(recent) - min(recent) < tolerance:
                break

    # After training, compute predictions for all nodes in this batch:
    # Build a tensor of size [batch_size, embedding_dim] for the new embeddings
    batch_embs = torch.stack(
        [paper_dict['paper'][int(n)] for n in random_sample],
        dim=0
    )  # shape [B, D]

    # Stack all venue embeddings into [V, D]
    venue_ids = sorted(paper_dict['venue'].keys())
    venue_tensor = torch.stack(
        [paper_dict['venue'][vid] for vid in venue_ids],
        dim=0
    )  # shape [V, D]

    # Compute pairwise squared distances: [B, V]
    diffs = batch_embs.unsqueeze(1) - venue_tensor.unsqueeze(0)  # [B, V, D]
    d2 = (diffs ** 2).sum(dim=2)                                  # [B, V]
    logits = torch.sigmoid(alpha - d2)                            # [B, V]
    probs = F.softmax(logits, dim=1)                              # [B, V]

    # For each of the B nodes, extract prediction and true rank
    for idx_in_batch, test_node in enumerate(random_sample):
        true_label = int(venue_value[int(test_node)].cpu().item())
        row_probs = probs[idx_in_batch]           # shape [V]
        sorted_probs, sorted_indices = torch.sort(row_probs, descending=True)
        ranked_venue_ids = [venue_ids[i] for i in sorted_indices.tolist()]

        if true_label in ranked_venue_ids:
            true_class_rank = ranked_venue_ids.index(true_label) + 1
        else:
            true_class_rank = -1

        predicted_node_id = ranked_venue_ids[0]
        predictions[int(test_node)] = (true_label, predicted_node_id, true_class_rank)

    # Recompute overall accuracy so far
    items = sorted(predictions.items())
    y_true = torch.tensor([tp[0] for _, tp in items], dtype=torch.long, device=device).view(-1, 1)
    y_pred = torch.tensor([tp[1] for _, tp in items], dtype=torch.long, device=device).view(-1, 1)
    evaluator = Evaluator(name='ogbn-mag')
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
    acc.append(result['acc'])
    # wandb.log({"Accuracy": result['acc']})
    print(f"Iteration {i + 1}/{num_iterations}, Accuracy: {result['acc']:.4f}")

    # Update remaining test nodes for next iteration
    # l_next is already a torch.Tensor; convert to a Python list of ints
    l_prev = l_next.tolist()

# Save the predictions dictionary
save_dir = 'dataset/ogbn_mag/processed/Predictions'
os.makedirs(save_dir, exist_ok=True)
torch.save(predictions, os.path.join(save_dir, f'pred_dict_{embedding_dim}.pt'))

# Filter paper_dict to only include papers for which predictions were made
filtered_paper_dict = {
    'paper': {k: v for k, v in paper_dict['paper'].items() if k in predictions},
    'venue': paper_dict['venue']
}

# Plot embeddings (sample 1000 if there are many)
plot_paper_venue_embeddings(
    venue_value=venue_value,
    embed_dict=filtered_paper_dict,
    sample_size=1000,
    filename='dataset/ogbn_mag/processed/Predictions/plot_valid_top_5',
    dim=embedding_dim,
    top=5
)

plot_paper_venue_embeddings(
    venue_value=venue_value,
    embed_dict=filtered_paper_dict,
    sample_size=1000,
    filename='dataset/ogbn_mag/processed/Predictions/plot_valid_top_1',
    dim=embedding_dim,
    top=1
)
