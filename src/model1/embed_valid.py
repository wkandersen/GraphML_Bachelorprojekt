import torch
import sys
import os
import gc
from Packages.mini_batches import mini_batches_code
# from Packages.loss_function import LossFunction
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid, data
import wandb
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss
import torch

class LossFunction:
    def __init__(self, alpha=1.0, eps=1e-8, use_regularization=False, lam=0.01):
        """
        Initialize the loss function with given parameters.
        
        Args:
            alpha (float): Scaling parameter for edge probability.
            eps (float): Small value to prevent log(0).
            use_regularization (bool): Whether to include Gaussian regularization.
        """
        self.alpha = alpha
        self.eps = eps
        self.use_regularization = use_regularization
        self.lam = lam

    def edge_probability(self, z_i, z_j):
        """Compute the probability of an edge existing between two embeddings."""
        dist_sq = torch.sum((z_i - z_j) ** 2, dim=1)  # Squared Euclidean distance (batch-wise)
        return 1 / (1 + torch.exp(-self.alpha + dist_sq))  # Logistic function, element-wise

    def link_loss(self, label, z_u, z_v):
        """Compute the loss for a single edge."""
        prob = self.edge_probability(z_u, z_v)  # Compute edge probabilities (batch-wise)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)  # Numerical stability

        # Compute the loss for each edge
        return label * torch.log(prob) + (1 - label) * torch.log(1 - prob)

    def compute_loss(self, z, datamatrix_tensor):
        """Compute the total loss for the dataset."""
        # Extract labels, u_idx, and v_idx in a vectorized way
        labels = datamatrix_tensor[:, 0].float()
        u_idx = datamatrix_tensor[:, 1].long()
        v_idx = datamatrix_tensor[:, 2].long()
        pv1_idx = datamatrix_tensor[:, 3].long()
        pv2_idx = datamatrix_tensor[:, 4].long()

        edge_entities = {
            0: 'paper',
            1: 'author',
            2: 'institution',
            3: 'field_of_study',
            4: 'venue'
        }

        # Get embeddings for u_idx and v_idx with validation
        z_u_list = []
        z_v_list = []
        new_labels = []
        missing = []

        for label, i_u, i_v, t_u, t_v in zip(labels, u_idx, v_idx, pv1_idx, pv2_idx):
            ent_u = edge_entities[t_u.item()]
            ent_v = edge_entities[t_v.item()]
            idx_u = i_u.item()
            idx_v = i_v.item()

            if idx_u in z[ent_u] and idx_v in z[ent_v]:
                z_u_list.append(z[ent_u][idx_u])
                z_v_list.append(z[ent_v][idx_v])
                new_labels.append(label)
            else:
                missing.append(idx_u)
                missing.append(idx_v)

        if not z_u_list or not z_v_list:
            raise ValueError("No valid pairs found for z_u and z_v.")

        z_u = torch.stack(z_u_list)
        z_v = torch.stack(z_v_list)
        labels = torch.tensor(new_labels, dtype=torch.float)



        # Compute link loss for all edges in the batch
        link_loss = self.link_loss(labels, z_u, z_v)  # shape (B,)

        # Mean loss over the batch
        loss = -torch.mean(link_loss)

        # Optionally add regularization
        if self.use_regularization:
            regularization = self.lam * torch.sum(z ** 2)
            loss += regularization

        return loss, set(missing)

venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device, weights_only=False)

learning_rate = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
num_iterations = 2
emb_dim = 2


check = torch.load('/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/checkpoint/checkpoint_iter_75_2_3_epoch_3.pt',map_location='cpu',weights_only=False)
paper_dict = check['collected_embeddings']

loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

unique_train = set(paper_c_paper_train.flatten().unique().tolist())
unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())

# Keep only validation nodes that do not appear in training edges
valid_exclusive = unique_valid - unique_train
l_prev = list(valid_exclusive)
predictions = {}

for i in range(num_iterations):
    mini_b = mini_batches_code(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'),data)
    dm, l_next, random_sample = mini_b.data_matrix()
    if len(dm) <= 1:
        continue
    dm = dm[dm[:,4]!=4]
    print(dm)

    mini_btrain1 = mini_batches_code(paper_c_paper_train, dm[:,2].tolist(), len(dm[:,2]), ('paper', 'cites', 'paper'),data)
    dmtrain1, ultrain1, random_sampletrain1 = mini_btrain1.data_matrix()

    test1 = dmtrain1[dmtrain1[:,4]!=4]
    test2 = test1[test1[:,0]==1]
    list_test = (test2[:,2].unique()).tolist() # list of all papers that train_papers are connected to

    concat_dm = torch.cat((dm,dmtrain1),0)

    rows = [[0, random_sample[0], test_item, 0, 0] for test_item in list_test]

    # Convert the list of rows to a tensor
    tensor_result = torch.tensor(rows)

    final = torch.cat((concat_dm,tensor_result),0)

    # Initialize a tensor of zeros with the same size as the tensors in paper_dict[i]
    # We'll accumulate the tensors in this tensor, and later divide by the count
    mean_tensor = None
    count = 0

    new_embedding = torch.nn.Parameter(torch.randn(2, requires_grad=True))


    # Now mean_tensor holds the element-wise mean of all tensors
    print(f"Element-wise mean tensor: {mean_tensor}")
    
    paper_dict['paper'][random_sample[0]] = new_embedding
    print(paper_dict['paper'][random_sample[0]])
    new_optimizer = torch.optim.Adam([paper_dict['paper'][random_sample[0]]], lr=learning_rate)

    # Training loop for multiple samples
    for epoch in range(num_epochs):
        new_optimizer.zero_grad()
        

        # temp_embed = torch.stack(list(paper_dict.values())) # SKAL ÆNDRES
        # Concatenate the embeddings
        loss,missing = loss_function.compute_loss(paper_dict, final)  # Compute loss

        # Backpropagation and optimization
        loss.backward()
        # print(new_embedding.grad)
        new_optimizer.step()

        # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    if len(missing) != 0:
        print(f'Missing: {missing}')

    print(paper_dict['paper'][random_sample[0]])
    alpha = 0.001
    logi_f = []

    for i in range(len(paper_dict['venue'])):
        dist = torch.norm(paper_dict['paper'][random_sample[0]] - paper_dict['venue'][i])**2  # Euclidean distance
        logi = 1 / (1 + torch.exp(alpha + dist))  # Logistic function
        logi_f.append((logi.item(), i))  # Store tuple (probability, node ID)

    # Separate values for softmax computation
    logits, node_ids = zip(*logi_f)  # Unzips into two lists

    # Convert logits to a tensor and apply softmax
    logi_f_tensor = torch.tensor(logits).to(device)
    softma = F.softmax(logi_f_tensor, dim=0)

    # Get the index of the highest probability
    high_prob_idx = torch.argmax(softma).item()

    # Get the corresponding node ID and its softmax probability
    predicted_node_id = node_ids[high_prob_idx]
    highest_prob_value = softma[high_prob_idx].item()
    predictions[random_sample[0]] = (int(venue_value[random_sample[0]].cpu().numpy()), predicted_node_id)

    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

# torch.save(predictions)
torch.save(predictions,f'dataset/ogbn_mag/processed/Predictions/pred_dict_{emb_dim}.pt')
print('finish')