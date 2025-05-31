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

paper_c_paper_train, paper_c_paper_valid, paper_c_paper_test, paper_c_paper = process_edges(
    data, ('paper', 'cites', 'paper'), nums_train, nums_valid, nums_test,mask_position='both')

paper_t_field_train, paper_t_field_valid, paper_t_field_test, paper_t_field = process_edges(
    data, ('paper', 'has_topic', 'field_of_study'), nums_train, nums_valid, nums_test,mask_position='source')

author_w_paper_train, author_w_paper_valid, author_w_paper_test, author_w_paper = process_edges(
    data, ('author', 'writes', 'paper'), nums_train, nums_valid, nums_test,mask_position='target')

nums_test_author = author_w_paper_test[0].unique().clone().detach()
nums_valid_author = author_w_paper_valid[0].unique().clone().detach()
nums_train_author = author_w_paper_train[0].unique().clone().detach()

author_a_institution_train, author_a_institution_valid, author_a_institution_test, author_a_institution = process_edges(
    data, ('author', 'affiliated_with', 'institution'), nums_train_author, nums_valid_author, nums_test_author,mask_position='source')

venue_value = {}
for idx, value in enumerate(data['y_dict']['paper']):
    venue_value[idx] = value

torch.save(venue_value, "dataset/ogbn_mag/processed/venue_value.pt")

# Extract unique filtered paper IDs for train, valid, test splits
nums_train_filtered = paper_c_paper_train.unique()
nums_valid_filtered = paper_c_paper_valid.unique()
nums_test_filtered = paper_c_paper_test.unique()

# Save these filtered IDs for use later in prep_data
torch.save(nums_train_filtered, "dataset/ogbn_mag/processed/nums_train_filtered.pt")
torch.save(nums_valid_filtered, "dataset/ogbn_mag/processed/nums_valid_filtered.pt")
torch.save(nums_test_filtered, "dataset/ogbn_mag/processed/nums_test_filtered.pt")
