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

import os
import torch

import torch

collected_embeddings = {
    'paper': {},
    'venue': {}
}

embedding_dim = 2
a = -1
b = -a
# Venue embeddings
embed_venue = torch.nn.Embedding(len(venues_values), embedding_dim+2)
torch.nn.init.uniform_(embed_venue.weight, a, b)

venue_id_to_idx = {venue_id.item(): idx for idx, venue_id in enumerate(venues_values)}

indices = torch.tensor([venue_id_to_idx[venue_id.item()] for venue_id in venues_values], dtype=torch.long)
embeddings_venue = embed_venue(indices)

for venue_id in venues_values:
    collected_embeddings['venue'][venue_id.item()] = embeddings_venue[venue_id_to_idx[venue_id.item()]]

print(f"Number of unique venues: {len(collected_embeddings['venue'])}")

# Paper embeddings
unique_paper_ids = torch.unique(paper_c_paper_train)
embed_paper = torch.nn.Embedding(len(unique_paper_ids), embedding_dim)
torch.nn.init.uniform_(embed_paper.weight, a, b)
paper_id_to_idx = {pid.item(): idx for idx, pid in enumerate(unique_paper_ids)}

indices = torch.tensor([paper_id_to_idx[pid.item()] for pid in paper_c_paper_train.flatten()], dtype=torch.long)
embeddings_paper = embed_paper(indices)

for pid, emb in zip(paper_c_paper_train.flatten(), embeddings_paper):
    collected_embeddings['paper'][pid.item()] = emb

# Save the combined embeddings dictionary (if it's populated)
if collected_embeddings:
    torch.save(collected_embeddings, f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_{b}_hybrid.pt")
    print("embeddings saved")


# # Load the collected embeddings dictionary
# collected_embeddings = torch.load(f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_{b}.pt")

# # Get the first 5 paper embeddings
# print("First 5 paper embeddings:")
# paper_embeddings = list(collected_embeddings['paper'].items())[:5]
# for paper_id, emb in paper_embeddings:
#     print(f"Paper ID: {paper_id}, Embedding: {emb}")

# # Get the first 5 venue embeddings
# print("\nFirst 5 venue embeddings:")
# venue_embeddings = list(collected_embeddings['venue'].items())[:5]
# for venue_id, emb in venue_embeddings:
#     print(f"Venue ID: {venue_id}, Embedding: {emb}")

