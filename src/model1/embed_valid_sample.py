import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.mini_batches import mini_batches_code
from Packages.loss_function import LossFunction
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

emb_matrix = torch.load("dataset/ogbn_mag/processed/hpc/emb_matrix_8_125_epoch.pt", map_location=device)
# paper_c_paper_valid = torch.load("dataset/ogbn_mag/processed/paper_c_paper_valid.pt", map_location=device)
data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

valid_dict = {}

# Get unique node IDs from both train and valid edges
unique_train = set(paper_c_paper_train.flatten().unique().tolist())
unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())

# Keep only validation nodes that do not appear in training edges
valid_exclusive = unique_valid - unique_train

# Initial list of nodes for iterations
l_prev = list(valid_exclusive)
num_iterations = int(len(l_prev)-1)

sample = 1


for i in range(num_iterations):
    print(f"Iteration {i+1}")

    # Generate mini-batches
    mini_b_new = mini_batches_code(paper_c_paper_valid, l_prev, sample, ('paper', 'cites', 'paper'),data)
    dm_new,l_next,remapped_datamatrix_tensor_new,random_sample = mini_b_new.node_mapping()

    dm_new = dm_new.to(device)
    remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new.to(device)

    new_datamatrix = dm_new[torch.all(dm_new[:, 4:] != 4, dim=1)]
    new_remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new[torch.all(remapped_datamatrix_tensor_new[:, 4:] != 4, dim=1)]

    loss_function = LossFunction(alpha=10, eps=1e-10, use_regularization=True, lam=0.001)
    for j in range(sample):

        new_embedding = torch.nn.Embedding(sample, 8).to(device)
        valid_dict[random_sample[j]] = new_embedding
        

    new_optimizer = torch.optim.Adam(new_embedding.parameters(), lr=0.001)

    num_epochs = 30
    
        # Training loop
    for epoch in range(num_epochs):
        new_optimizer.zero_grad()

        # Concatenate the embeddings
        temp_embed = torch.cat([emb_matrix, new_embedding.weight], dim=0)
        types = new_datamatrix[:, 3:]
        loss = loss_function.compute_loss(temp_embed, new_remapped_datamatrix_tensor_new[:, :3])  # Compute loss
        
        # Backpropagation and optimization
        loss.backward()
        new_optimizer.step()

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Update node list for the next iteration
    l_prev = l_next

    # valid_dict[random_sample[0]] = new_embedding.weight.detach().cpu().clone()

    # Cleanup
    if (i + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

torch.save(valid_dict, "dataset/ogbn_mag/processed/hpc/valid_dict_8.pt")

print('embed_valid done')
        