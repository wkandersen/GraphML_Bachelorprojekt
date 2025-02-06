from torch_geometric.datasets import OGB_MAG

dataset = OGB_MAG(root="dataset/")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object

