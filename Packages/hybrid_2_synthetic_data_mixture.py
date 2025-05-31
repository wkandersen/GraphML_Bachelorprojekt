import numpy as np
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torch
import random
from collections import Counter
from Packages.plot_embeddings import set_seed

set_seed(69)
embedding_dim = 2
a = -1
b = -a

n_samples = 300
# 1) Define our four quadrant means and identical isotropic covariances
mean = 0.25
means = np.array([
    [ mean,  mean],   # Q1: x∈(0,1), y∈(0,1)
    [-mean,  mean],   # Q2: x∈(-1,0),y∈(0,1)
    [-mean, -mean],   # Q3: x∈(-1,0),y∈(-1,0)
    [ mean, -mean],   # Q4: x∈(0,1), y∈(-1,0)
])
covariance = np.eye(2) * (0.1 ** 2)  # std = 0.2
covariances = np.stack([covariance] * 4, axis=0)

# 2) Build a “fake” GMM by directly assigning its parameters
gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.weights_ = np.ones(4) / 4
gmm.means_ = means
gmm.covariances_ = covariances

# compute precisions_cholesky_ for full covariance
precisions_cholesky = np.zeros_like(covariances)
for i in range(4):
    # precision = inv(covariance)
    P = np.linalg.inv(covariances[i])
    # cholesky of precision
    precisions_cholesky[i] = np.linalg.cholesky(P)
gmm.precisions_cholesky_ = precisions_cholesky

# 3) Oversample from the mixture and carve out exactly 50 per component
#    (we draw 1000 points so that each component will almost surely have ≥50)
samples, labels = gmm.sample(n_samples*4)

selected = []
for comp in range(4):
    idx = np.where(labels == comp)[0][:n_samples // 4]
    selected.append(samples[idx])
emb_data = np.vstack(selected)  # shape (200,2)

# 4) Stuff into a torch.nn.Embedding
emb = nn.Embedding(n_samples, 2)
with torch.no_grad():
    emb.weight.copy_(torch.from_numpy(emb_data).float())

print(emb.weight.shape)  # torch.Size([200, 2])

# embeddings are defined in emb
embeddings = emb.weight.detach()
# labels are the component labels
labels = gmm.predict(embeddings)
# mean = 0.2
venue_embeds = np.array([
    [0, 0],
    [0, -mean],
    [-mean, 0],
    [0, mean],
    [mean, 0],
    [-mean, -mean],
    [-mean, mean],
    [mean, -mean],
    [mean, mean]
])
# create a new embedding layer
venue_emb = nn.Embedding(9, 2)
# copy the weights
with torch.no_grad():
    venue_emb.weight.copy_(torch.from_numpy(venue_embeds).float())

# show plot of the embeddings with venue embeddings in a different color
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.scatter(embeddings[labels == i, 0], embeddings[labels == i, 1], label=f'Component {i+1}', alpha=0.5)
plt.scatter(venue_emb.weight[:, 0].detach().numpy(), venue_emb.weight[:, 1].detach().numpy(), label='Venue Embeddings', color='black', alpha=0.5)
plt.title('GMM Embeddings with Venue Embeddings')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()

def edge_probability(z_i, z_j, alpha=2):
    """Compute the probability of an edge existing between two embeddings."""
    dist_sq = torch.sum((z_i - z_j) ** 2)  # Squared Euclidean distance (batch-wise)
    # return 1 / (1 + torch.exp(-self.alpha + dist_sq))  # Logistic function, element-wise
    return torch.sigmoid(-dist_sq + alpha)

venue_embeddings = venue_emb.weight.detach()
venues_values = {}
max_indices = []
for i in range(embeddings.shape[0]): 
    venue_dist = []
    for j in range(venue_embeddings.shape[0]):
        venue_dist.append((edge_probability(embeddings[i],venue_embeddings[j]).item(),j))

    max_prob, max_index = max(venue_dist, key=lambda x: x[0])
    venues_values[i] = torch.tensor(max_index)
    max_indices.append([max_index])

min_index_tensor = torch.tensor(max_indices)

# Now build the dictionary
data = {
    'y_dict': {
        'paper': min_index_tensor
    }
}

train,valid,test = 0.7,0.2,0.10
num_edges_train = (3879968/623568)*(embeddings.shape[0]*train) #edges divided by nodes in training timed with nodes in synthetic
num_edges_valid = (847520/330812)*(embeddings.shape[0]*valid)
num_edges_test = (688783/275074)*(embeddings.shape[0]*test)

num_nodes = embeddings.shape[0]
threshold = 0.8

edges = []
added = set()

# To ensure fair distribution, randomly sample node pairs
attempts = 0
max_attempts = num_edges_train * 10  # prevent infinite loop if threshold too high

while len(edges) < num_edges_train and attempts < max_attempts:
    i = random.randint(0, num_nodes - 1)
    j = random.randint(0, num_nodes - 1)

    if i == j:
        attempts += 1
        continue

    key = (i, j)
    if key in added:
        attempts += 1
        continue

    prob = edge_probability(embeddings[i], embeddings[j]).item()

    if prob > threshold:
        edges.append(key)
        added.add(key)

    attempts += 1

# Convert to tensor
edge_index = torch.tensor(edges).T  # shape: (2, num_edges_train)
sorted_indices = edge_index[0].argsort()
paper_c_paper_train = edge_index[:, sorted_indices]

def edges_set(num_edges,range_i0,range_i1,range_j0,range_j1,max_attempts=1000):
    edges = []
    attempts = 0
    added = set()
    while len(edges) < num_edges and attempts < max_attempts:
        i = random.randint(range_i0,range_i1)
        j = random.randint(range_j0,range_j1)
        if i == j:
            attempts += 1
            continue

        key = (i, j)
        if key in added:
            attempts += 1
            continue

        prob = edge_probability(embeddings[i], embeddings[j]).item()

        if prob > threshold:
            edges.append(key)
            added.add(key)

        attempts += 1

    edge_index = torch.tensor(edges).T  # shape: (2, num_edges_train)
    sorted_indices = edge_index[0].argsort()
    paper_edges = edge_index[:, sorted_indices]

    return paper_edges


i0train,i1train,j0train,j1train = 0,int(num_nodes*train),0,int(num_nodes*train)
paper_c_paper_train=edges_set(num_edges=num_edges_train,range_i0=i0train,range_i1=i1train,range_j0=j0train,range_j1=j1train)

i0valid,i1valid,j0valid,j1valid = int(num_nodes*train)+1,int(num_nodes*train)+int(num_nodes*valid),0,int(num_nodes*train)+int(num_nodes*valid)
paper_c_paper_valid=edges_set(num_edges=num_edges_valid,range_i0=i0valid,range_i1=i1valid,range_j0=j0valid,range_j1=j1valid)

i0test,i1test,j0test,j1test = int(num_nodes*train)+int(num_nodes*valid)+1,num_nodes-1,0,num_nodes-1
paper_c_paper_test=edges_set(num_edges=num_edges_test,range_i0=i0test,range_i1=i1test,range_j0=j0test,range_j1=j1test)

print(i0valid,i1valid)

print(paper_c_paper_test.shape,paper_c_paper_train.shape,paper_c_paper_valid.shape)

collected_embeddings = {
    'paper': {},
    'venue': {}
}
# Print how many values are assigned to each venue key
venue_counts = Counter([v.item() for v in venues_values.values()])
print("Venue assignment counts:", dict(venue_counts))

# Get unique paper indices in the validation set
unique_papers = np.unique(paper_c_paper_valid[0].numpy())
# Map each paper to its venue using venues_values
venue_assignments = [venues_values[int(pid)].item() for pid in unique_papers]
# Count how many papers are assigned to each venue
venue_count = Counter(venue_assignments)
print("Venue assignment counts for validation set:", dict(venue_count))


unique_papers = np.unique(paper_c_paper_test[0].numpy())
# Map each paper to its venue using venues_values
venue_assignments = [venues_values[int(pid)].item() for pid in unique_papers]
# Count how many papers are assigned to each venue
venue_count = Counter(venue_assignments)
print("Venue assignment counts for test set:", dict(venue_count))
# print(venues_values)

unique_papers = np.unique(paper_c_paper_train[0].numpy())
# Map each paper to its venue using venues_values
venue_assignments = [venues_values[int(pid)].item() for pid in unique_papers]
# Count how many papers are assigned to each venue
venue_count = Counter(venue_assignments)
print("Venue assignment counts for train set:", dict(venue_count))
# print(venues_values)



# Get unique venue IDs assigned to papers
unique_venues = set([v.item() for v in venues_values.values()])  # should be 9 unique venues

# Create venue embedding layer for those unique venues
venue_embed = torch.nn.Embedding(len(unique_venues), embedding_dim + 2)

# Then save embeddings indexed by venue IDs, not paper IDs
for venue_id in unique_venues:
    collected_embeddings['venue'][venue_id] = venue_embed.weight[venue_id].detach().clone()



# Paper embeddings
unique_paper_ids = torch.unique(paper_c_paper_train)
paper_embed = torch.nn.Embedding(len(unique_paper_ids), embedding_dim)
torch.nn.init.uniform_(paper_embed.weight, a, b)
paper_id_to_idx = {pid.item(): idx for idx, pid in enumerate(unique_paper_ids)}

indices = torch.tensor([paper_id_to_idx[pid.item()] for pid in paper_c_paper_train.flatten()], dtype=torch.long)
embeddings = paper_embed(indices)

for pid, emb in zip(paper_c_paper_train.flatten(), embeddings):
    collected_embeddings['paper'][pid.item()] = emb.detach().clone()

dict2 = {}
for pid in unique_paper_ids:
    emb = torch.nn.Embedding(1, 2)
    torch.nn.init.uniform_(emb.weight, a=0.0, b=1.0)  # Optional init
    dict2[pid.item()] = emb.weight.detach().clone().squeeze(0) 

save_path = f"src/hybrid/MLP/embeddings/save_dim{embedding_dim}_b{b}.pt"
if collected_embeddings:
    save = {
        'collected_embeddings': collected_embeddings,
        'paper_c_paper_train': paper_c_paper_train,
        'paper_c_paper_valid': paper_c_paper_valid,
        'paper_c_paper_test': paper_c_paper_test,
        'data': data,
        'venue_value': venues_values,
        'emb2d': dict2
    }
    torch.save(save, save_path)
    print("embeddings saved")

for k, v in collected_embeddings['paper'].items():
    assert v.shape[0] == 2, f"Paper {k} has shape {v.shape} instead of [2]"

save = torch.load(save_path)

print(save['collected_embeddings']['venue'].keys())


# print("collected_embeddings:", collected_embeddings["venue"])
# print("collected_embeddings length:", collected_embeddings['paper'])
# print("colletec_embeddings length:", len(collected_embeddings['venue']))
# 

# Load the collected embeddings dictionary
# collected_embeddings = torch.load(f"src/model1/syntetic_data/embed_dict/collected_embeddings_{embedding_dim}_spread_{b}_synt_new.pt")

