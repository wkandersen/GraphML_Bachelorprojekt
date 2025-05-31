import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def process_edges(data, edge_type, nums_train, nums_valid, nums_test, mask_position):
    edge_data = data.edge_index_dict[edge_type]

    # If check_both is True, check both parts of the edge
    if mask_position == 'both':
        mask_train = torch.isin(edge_data[0], nums_train) | torch.isin(edge_data[1], nums_train)
        mask_valid = torch.isin(edge_data[0], nums_valid) | torch.isin(edge_data[1], nums_valid)
        mask_test = torch.isin(edge_data[0], nums_test) | torch.isin(edge_data[1], nums_test)
    elif mask_position == 'source':
        mask_train = torch.isin(edge_data[0], nums_train)
        mask_valid = torch.isin(edge_data[0], nums_valid)
        mask_test = torch.isin(edge_data[0], nums_test)
    elif mask_position == 'target':
        mask_train = torch.isin(edge_data[1], nums_train)
        mask_valid = torch.isin(edge_data[1], nums_valid)
        mask_test = torch.isin(edge_data[1], nums_test)
    else:
        raise ValueError('mask_position must be one of "both", "source", or "target"')
    
    # Combine the conditions into a single mask that selects only the train edges
    mask_train_done = mask_train & ~mask_valid & ~mask_test
    mask_valid_done = mask_valid & ~mask_test

    edge_train = edge_data.clone()[:, mask_train_done]
    edge_valid = edge_data.clone()[:, mask_valid_done]
    edge_test = edge_data.clone()[:, mask_test]

    return edge_train, edge_valid, edge_test, edge_data

data_train = pd.read_csv(r'dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
data_valid = pd.read_csv(r'dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
data_test = pd.read_csv(r'dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

nums_valid = torch.tensor(data_valid[0])
nums_test = torch.tensor(data_test[0])
nums_train = torch.tensor(data_train[0])

paper_c_paper_train, paper_c_paper_valid_before, paper_c_paper_test_before, paper_c_paper_before = process_edges(
    data, ('paper', 'cites', 'paper'), nums_train, nums_valid, nums_test,mask_position='both')

venue_value = {}
for idx, value in enumerate(data['y_dict']['paper']):
    venue_value[idx] = value

print("Train test:", len(
    set(paper_c_paper_test_before[0].unique().tolist())
      - set(paper_c_paper_train.flatten().unique().tolist())
      - set(paper_c_paper_valid_before[0].unique().tolist())
))
only_in_train = set(paper_c_paper_train.flatten().unique().tolist())
only_in_valid = set(paper_c_paper_valid_before[0].unique().tolist()) - set(paper_c_paper_train.flatten().unique().tolist())
paper_c_paper_valid = paper_c_paper_valid_before[:, torch.isin(paper_c_paper_valid_before[0], torch.tensor(list(only_in_valid)))]

only_in_test = set(paper_c_paper_test_before[0].unique().tolist()) - set(paper_c_paper_train.flatten().unique().tolist()) - set(paper_c_paper_valid_before[0].unique().tolist())

paper_c_paper_test = paper_c_paper_test_before[:, torch.isin(paper_c_paper_test_before[0], torch.tensor(list(only_in_test)))]

# Save these filtered IDs for use later in prep_data
torch.save(only_in_train, "dataset/ogbn_mag/processed/nums_train_filtered.pt")
torch.save(only_in_valid, "dataset/ogbn_mag/processed/nums_valid_filtered.pt")
torch.save(only_in_test, "dataset/ogbn_mag/processed/nums_test_filtered.pt")

# check for overlap
print("Overlap between train and valid:", len(
    set(paper_c_paper_train.flatten().unique().tolist())
      & set(paper_c_paper_valid[0].unique().tolist())
))
print("Overlap between train and test:", len(
    set(paper_c_paper_train.flatten().unique().tolist())
      & set(paper_c_paper_test[0].unique().tolist())
))
print("Overlap between valid and test:", len(
    set(paper_c_paper_valid[0].unique().tolist())
      & set(paper_c_paper_test[0].unique().tolist())
))