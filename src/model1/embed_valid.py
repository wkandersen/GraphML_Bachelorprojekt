import torch
import sys
import os
import gc
from Packages.mini_batches import mini_batches_code
from Packages.loss_function import LossFunction
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid, data
import wandb
import torch.nn as nn

paper_dict = torch.load('/mnt/c/Users/Bruger/Desktop/Bachelor/GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/hpc/collected_embeddings_2.pt')

learning_rate = 0.001
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
num_iterations = 10
emb_dim = 2

loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

unique_train = set(paper_c_paper_train.flatten().unique().tolist())
unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())

# Keep only validation nodes that do not appear in training edges
valid_exclusive = unique_valid - unique_train
l_prev = list(valid_exclusive)

for i in range(num_iterations):
    mini_b = mini_batches_code(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'),data)
    dm, l_next, random_sample = mini_b.data_matrix()
    print(dm)
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

if len(dm[:, 2].tolist()) < 2:
    new_embedding = nn.Embedding(1,emb_dim)
else:
    # Iterate over the elements in dm[:, 2]
    for i in dm[:, 2].tolist():
        tensor_value = paper_dict['paper'][i]  # Access the tensor from paper_dict
        if mean_tensor is None:
            mean_tensor = tensor_value.clone()  # Initialize the mean_tensor with the first tensor
        else:
            mean_tensor += tensor_value  # Accumulate the sum of tensors
        count += 1  # Increment the count of tensors

    # Compute the element-wise mean by dividing the accumulated sum by count
    mean_tensor /= count
    
    new_embedding = nn.Parameter(mean_tensor)

    # Now mean_tensor holds the element-wise mean of all tensors
    print(f"Element-wise mean tensor: {mean_tensor}")

    paper_dict['paper'][random_sample[0]] = torch.nn.Parameter(new_embedding.clone().detach())
    print(paper_dict['paper'][random_sample[0]])
    new_optimizer = torch.optim.Adam([paper_dict['paper'][random_sample[0]]], lr=learning_rate)

    # Training loop for multiple samples
    for epoch in range(num_epochs):
        new_optimizer.zero_grad()

        # temp_embed = torch.stack(list(paper_dict.values())) # SKAL ÆNDRES
        # Concatenate the embeddings
        loss = loss_function.compute_loss(paper_dict, final)  # Compute loss

        # Backpropagation and optimization
        loss.backward()
        # print(new_embedding.grad)
        new_optimizer.step()

        # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    print(paper_dict['paper'][random_sample[0]])

    l_prev = l_next

    # Cleanup
    if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
        import gc
        gc.collect()
        torch.cuda.empty_cache()