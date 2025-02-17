from ogb.nodeproppred import PygNodePropPredDataset

# Load the dataset
dataset = PygNodePropPredDataset(name = "ogbn-mag")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object


print(graph.num_nodes_dict)
