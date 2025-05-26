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

set_seed(42)

### paper_c_paper_train
num_papers = 200     # total number of papers (IDs from 0 to 749999)
num_edges = 300        # number of citation edges
embedding_dim = 2
a = -1
b = -a

# Randomly generate citing and cited paper IDs
citing = torch.randint(0, num_papers, (num_edges,))
cited = torch.randint(0, num_papers, (num_edges,))

# Stack to form a 2 x num_edges tensor
paper_c_paper_train = torch.stack([citing, cited])

### data and venue_value

# Generate a tensor of N random integers between 0 and 50
num_values = num_papers  # You can change this number
random_values = torch.randint(0, 51, (num_values, 1))

# Create the dictionary
data = {
    'y_dict': {
        'paper': random_values
    }
}

tensor_values = data['y_dict']['paper']
venue_value = {i: tensor_values[i] for i in range(tensor_values.size(0))}

### data and venue_value

# Generate a tensor of N random integers between 0 and 50
num_values = num_papers  # You can change this number
random_values = torch.randint(0, 51, (num_values, 1))

# Create the dictionary
data = {
    'y_dict': {
        'paper': random_values
    }
}

tensor_values = data['y_dict']['paper']
venue_value = {i: tensor_values[i] for i in range(tensor_values.size(0))}

venues_values = torch.unique(data['y_dict']['paper'])

collected_embeddings = {
    'paper': {},
    'venue': {}
}
# Venue embeddings
embed = torch.nn.Embedding(len(venues_values), embedding_dim)
torch.nn.init.uniform_(embed.weight, a, b)

venue_id_to_idx = {venue_id.item(): idx for idx, venue_id in enumerate(venues_values)}

indices = torch.tensor([venue_id_to_idx[venue_id.item()] for venue_id in venues_values], dtype=torch.long)
embeddings = embed(indices)

for venue_id in venues_values:
    collected_embeddings['venue'][venue_id.item()] = embeddings[venue_id_to_idx[venue_id.item()]]

# Paper embeddings
unique_paper_ids = torch.unique(paper_c_paper_train)
embed = torch.nn.Embedding(len(unique_paper_ids), embedding_dim)
torch.nn.init.uniform_(embed.weight, a, b)
paper_id_to_idx = {pid.item(): idx for idx, pid in enumerate(unique_paper_ids)}

indices = torch.tensor([paper_id_to_idx[pid.item()] for pid in paper_c_paper_train.flatten()], dtype=torch.long)
embeddings = embed(indices)

for pid, emb in zip(paper_c_paper_train.flatten(), embeddings):
    collected_embeddings['paper'][pid.item()] = emb.detach().clone()

if collected_embeddings:
    torch.save(collected_embeddings, f"/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_{b}_synt.pt")
    print("embeddings saved")


# Load the collected embeddings dictionary
collected_embeddings = torch.load(f"/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/collected_embeddings_{embedding_dim}_spread_{b}_synt.pt")

### paper_c_paper_test
num_papers_test = 100  # Number of test papers
num_edges_test = 150   # Number of citation edges in test

# Generate citing and cited paper IDs for test
citing_test = torch.randint(0, num_papers_test, (num_edges_test,))
cited_test = torch.randint(0, num_papers_test, (num_edges_test,))
paper_c_paper_test = torch.stack([citing_test, cited_test])

### test data and venue_value_test

# Reuse the same venue distribution from training
venues_train = torch.unique(data['y_dict']['paper'])

# Randomly assign a venue to each test paper (using train venues)
venue_indices = torch.randint(0, len(venues_train), (num_papers_test,))
random_venues_for_test = venues_train[venue_indices]

# Create the test data dictionary
test_data = {
    'y_dict': {
        'paper': random_venues_for_test
    }
}

# Create the venue_value_test dictionary that maps test paper ID to venue
tensor_values_test = test_data['y_dict']['paper']
venue_value_test = {i: tensor_values_test[i] for i in range(tensor_values_test.size(0))}

# (Optional) Print to confirm
print("Unique test venues:", torch.unique(tensor_values_test))
print("Total test papers:", len(venue_value_test))
