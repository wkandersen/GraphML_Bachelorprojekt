import pandas as pd
import torch
import os
# Set working directory

try:
    data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
    data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
    data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)
except FileNotFoundError:
    os.chdir("..")
    data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
    data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
    data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)

data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

# Extract edges for "paper" -> "cites" -> "paper"
paper_c_paper = data.edge_index_dict[('paper', 'cites', 'paper')]

# Unique paper IDs to keep (Ensure it's a PyTorch tensor)
nums_valid = torch.tensor(data_valid[0])
nums_test = torch.tensor(data_test[0])
nums_train = torch.tensor(data_train[0])

mask_train = torch.isin(paper_c_paper[0], nums_train) | torch.isin(paper_c_paper[1], nums_train)
mask_valid = torch.isin(paper_c_paper[0], nums_valid) | torch.isin(paper_c_paper[1], nums_valid)
mask_test = torch.isin(paper_c_paper[0], nums_test) | torch.isin(paper_c_paper[1], nums_test)

paper_c_paper_train = paper_c_paper.clone()
paper_c_paper_valid = paper_c_paper.clone()
paper_c_paper_test = paper_c_paper.clone()

# Combine the conditions into a single mask that selects only the train edges
mask_train_done = mask_train & ~mask_valid & ~mask_test
mask_valid_done = mask_valid & ~mask_test

# Apply the combined mask to paper_c_paper_train
paper_c_paper_train = paper_c_paper_train[:, mask_train_done]
paper_c_paper_valid = paper_c_paper_valid[:, mask_valid_done]
paper_c_paper_test = paper_c_paper_test[:, mask_test]

#Venues
venues_values = torch.unique(data['y_dict']['paper'])

len(paper_c_paper_train[1]) + len(paper_c_paper_valid[1]) + len(paper_c_paper_test[1]), paper_c_paper.shape[1]

embedding_dim = 4

if not os.path.exists(f"dataset/ogbn_mag/processed/venue_embeddings_{embedding_dim}.pt"):
    venue_embeddings = {}

    embed = torch.nn.Embedding(len(venues_values), embedding_dim)

    venue_id_to_idx = {venue_id.item(): idx for idx, venue_id in enumerate(venues_values)}

    indices = torch.tensor([venue_id_to_idx[venue_id.item()] for venue_id in venues_values], dtype=torch.long)

    embeddings = embed(indices)

    venue_embeddings = {venue_id.item(): embeddings[venue_id_to_idx[venue_id.item()]] for venue_id in venues_values}

    # Save the embeddings to a file
    torch.save(venue_embeddings, f"dataset/ogbn_mag/processed/venue_embeddings_{embedding_dim}.pt")





if not os.path.exists(f"dataset/ogbn_mag/processed/paper_embeddings_{embedding_dim}.pt"):
    paper_embeddings = {}

    # Get unique paper IDs
    unique_paper_ids = torch.unique(paper_c_paper_train)

    # Define the embedding layer (one embedding per unique paper)
    embed = torch.nn.Embedding(len(unique_paper_ids), embedding_dim)

    # Create a mapping: paper ID → index in embedding layer
    paper_id_to_idx = {pid.item(): idx for idx, pid in enumerate(unique_paper_ids)}

    # Convert paper_c_paper_train to indices using vectorized operations
    indices = torch.tensor([paper_id_to_idx[pid.item()] for pid in paper_c_paper_train.flatten()])

    # Compute embeddings
    embeddings = embed(indices)

    # Convert to dictionary with original paper IDs
    paper_embeddings = {pid.item(): emb for pid, emb in zip(paper_c_paper_train.flatten(), embeddings)}

    torch.save(paper_embeddings, f"dataset/ogbn_mag/processed/paper_embeddings_{embedding_dim}.pt")


# paper = torch.load("dataset/ogbn_mag/processed/hpc/paper_dict_8.pt")
# venue = torch.load("dataset/ogbn_mag/processed/hpc/venue_dict_8.pt")


# print(paper[53719])
# print(venue)

# for key, tensor in paper.items():
#     if tensor.shape[0] != 8:
#         print(f"Paper {key} has shape {tensor.shape}")

# for key, tensor in venue.items():
#     if tensor.shape[0] != 8:
#         print(f"Venue {key} has shape {tensor.shape}")


# all_embeddings = list(paper.values()) + list(venue.values())
# for i, tensor in enumerate(all_embeddings):
#     if tensor.shape[0] != 8:
#         print(f"Entry {i} has shape {tensor.shape}")
# print(f"Total entries: {len(all_embeddings)}")
