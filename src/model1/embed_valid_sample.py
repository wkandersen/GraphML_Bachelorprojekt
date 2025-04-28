import torch
import sys
import os
import gc
from Packages.mini_batches import mini_batches_code
from Packages.loss_function import LossFunction
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid
import wandb

class EmbeddingTrainer_valid:
    def __init__(self, emb_matrix, embedding_dim=4, num_epochs=125, samples=1, learning_rate=0.01, alpha=1, eps=1e-10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.samples = samples  # Now accepts multiple samples
        self.learning_rate = learning_rate
        self.reg_alpha = alpha
        self.reg_eps = eps

        # self.emb_matrix = torch.load(f"dataset/ogbn_mag/processed/hpc/emb_matrix_{self.embedding_dim}_{self.num_epochs}_epoch.pt", map_location=self.device)
        self.emb_matrix = emb_matrix
        self.data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

        # Get unique node IDs from both train and valid edges
        self.unique_train = set(paper_c_paper_train.flatten().unique().tolist())
        self.unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())

        # Keep only validation nodes that do not appear in training edges
        self.valid_exclusive = self.unique_valid - self.unique_train

        self.valid_dict = {}

    def train(self):
        # Initial list of nodes for iterations
        l_prev = list(self.valid_exclusive)
        num_iterations = int(len(l_prev)/self.samples - 1)
        # num_iterations = 1

        for i in range(num_iterations):
            print(f"Iteration {i + 1}")

            # Generate mini-batches
            mini_b_new = mini_batches_code(paper_c_paper_valid, l_prev, self.samples, ('paper', 'cites', 'paper'), self.data)
            dm_new, l_next, remapped_datamatrix_tensor_new, random_sample = mini_b_new.node_mapping()

            dm_new = dm_new.to(self.device)
            remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new.to(self.device)

            new_datamatrix = dm_new[torch.all(dm_new[:, 4:] != 4, dim=1)]
            new_remapped_datamatrix_tensor_new = remapped_datamatrix_tensor_new[torch.all(remapped_datamatrix_tensor_new[:, 4:] != 4, dim=1)]

            loss_function = LossFunction(alpha=self.reg_alpha, eps=self.reg_eps, use_regularization=True)

            new_embedding = torch.nn.Embedding(self.samples, 2).to(self.device)

            new_optimizer = torch.optim.Adam(new_embedding.parameters(), lr=self.learning_rate)

            # Training loop for multiple samples
            for epoch in range(self.num_epochs):
                new_optimizer.zero_grad()

                # Concatenate the embeddings
                temp_embed = torch.cat([self.emb_matrix, new_embedding.weight], dim=0)
                types = new_datamatrix[:, 3:]
                loss = loss_function.compute_loss(temp_embed, new_remapped_datamatrix_tensor_new[:, :3])  # Compute loss

                # Backpropagation and optimization
                loss.backward()
                new_optimizer.step()

                wandb.log({"epoch_loss_valid": loss.item(), "epoch": epoch})

                # Print loss every 10 epochs
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

            # Update node list for the next iteration
            l_prev = l_next

            # Store the embeddings for each sample
            for sample_idx in range(self.samples):
                self.valid_dict[random_sample[sample_idx]] = new_embedding.weight.detach().cpu().clone()

            # Cleanup
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # # Save the final embeddings
        # torch.save(self.valid_dict, f"dataset/ogbn_mag/processed/hpc/valid_dict_{self.embedding_dim}_{self.num_epochs}_epoch.pt")
        print('embed_valid done')
        return self.valid_dict,loss

# # Example usage with more than one sample:
# trainer = EmbeddingTrainer(embedding_dim=4, num_epochs=30, samples=5)  # Using 5 samples
# trainer.train()
