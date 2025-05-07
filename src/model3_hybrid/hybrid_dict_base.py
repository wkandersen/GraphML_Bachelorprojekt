# imports
import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")
embedding_dim = 2
# Load initial embeddings
embed_dict = torch.load(f"dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}.pt", map_location=device)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)
X = data.x_dict[('paper')]

hybrid_dict = {}

for key in embed_dict:
    hybrid_dict[key] = {}

    if key == 'venue':
        # Direct copy of venue embeddings
        for idx, embedding in embed_dict[key].items():
            hybrid_dict[key][idx] = embedding  # no clone, no concat
    else:
                # Concatenate embed with X and make it a leaf again
        for idx, embedding in embed_dict['paper'].items():
            hybrid_dict['paper'][idx] = torch.cat((embed_dict['paper'][idx], X[idx]), -1)

torch.save(hybrid_dict, f"dataset/ogbn_mag/processed/hybrid_dict_{embedding_dim}.pt")