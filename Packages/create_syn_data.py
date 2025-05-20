import torch
### paper_c_paper_train
num_papers = 200     # total number of papers (IDs from 0 to 749999)
num_edges = 1000        # number of citation edges

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

embedding_dim = 2
a = -100
b = -a
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