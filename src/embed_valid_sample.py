import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.mini_batches import mini_batches_code
from Packages.loss_function import LossFunction

emb_matrix = torch.load("dataset/ogbn_mag/processed/emb_matrix.pt")
paper_c_paper_valid = torch.load("dataset/ogbn_mag/processed/paper_c_paper_valid.pt")


sample = 1
mini_b_new = mini_batches_code(paper_c_paper_valid, list(paper_c_paper_valid[0].unique().numpy()), 1, ('paper', 'cites', 'paper'))
dm_new,l_new,remapped_datamatrix_tensor_new = mini_b_new.node_mapping()
print(dm_new)
new_datamatrix = dm_new[torch.all(dm_new[:, 4:] != 4, dim=1)]
new_remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new[torch.all(remapped_datamatrix_tensor_new[:, 4:] != 4, dim=1)]
print(new_remapped_datamatrix_tensor_new)

loss_function = LossFunction(alpha=1, eps=1e-10, use_regularization=True)

new_embedding = torch.nn.Embedding(sample, 2)
print(new_embedding.weight)

new_optimizer = torch.optim.Adam(new_embedding.parameters(), lr=0.01)

num_epochs = 20


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


        