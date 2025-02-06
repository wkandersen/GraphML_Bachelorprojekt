import torch
from torch_geometric.data import Data
import os

data = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt")

# Check the contents
print(data)

print("Node Features:", data.x)
