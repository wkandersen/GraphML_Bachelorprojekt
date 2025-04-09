import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.loss_function import LossFunction
import wandb

class NodeEmbeddingTrainer:
    def __init__(self, dm, remapped_datamatrix_tensor, paper_dict, venue_dict, embedding_dim=2, num_epochs=10, lr=0.01, alpha=1,eps=1e-10,lam=0.01):
        # Initialize input data, parameters, and setup
        self.dm = dm
        self.remapped_datamatrix_tensor = remapped_datamatrix_tensor
        self.paper_dict = paper_dict
        self.venue_dict = venue_dict
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.alpha = alpha
        self.lam = lam
        self.eps = eps

        # Process data
        self.dm1 = dm[dm[:, 4] != 4]
        self.dm2 = dm[dm[:, 4] == 4]

        # Get node indices
        self.specific_papernode_indices = torch.cat([torch.unique(self.dm1[:, 1]), torch.unique(self.dm1[:, 2])], dim=0)
        self.specific_venuenode_indices = torch.unique(self.dm2[:, 2], dim=0)

        # Create embeddings
        self.papernode_embeddings = torch.nn.Embedding(len(self.specific_papernode_indices), self.embedding_dim)
        self.venuenode_embeddings = torch.nn.Embedding(len(self.specific_venuenode_indices), self.embedding_dim)

        # Optimizers
        self.paper_optimizer = torch.optim.Adam(self.papernode_embeddings.parameters(), lr=self.lr)
        self.venue_optimizer = torch.optim.Adam(self.venuenode_embeddings.parameters(), lr=self.lr)

        # Loss function (assumed to be defined elsewhere)
        self.loss_function = LossFunction(alpha=self.alpha, eps=self.eps, use_regularization=True, lam=self.lam)

    def train(self):
        venue_dict = self.venue_dict
        paper_dict = self.paper_dict
        # Training loop
        for epoch in range(self.num_epochs):
            self.paper_optimizer.zero_grad()
            self.venue_optimizer.zero_grad()

            # Concatenate the embeddings
            z = torch.cat((self.papernode_embeddings.weight, self.venuenode_embeddings.weight), dim=0)
            # types = self.dm[:, 3:]
            loss = self.loss_function.compute_loss(z, self.remapped_datamatrix_tensor[:, :3])  # Compute loss
            
            # Backpropagation and optimization
            loss.backward()
            self.paper_optimizer.step()
            self.venue_optimizer.step()

            # log to wandb
            wandb.log({"epoch_loss": loss.item(), "epoch": epoch})

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        print(self.specific_venuenode_indices)

        for idx, node in enumerate(self.specific_papernode_indices):
            paper_dict[int(node)] = self.papernode_embeddings.weight[idx].detach().cpu().clone()


        for idx, node in enumerate(self.specific_venuenode_indices):
            venue_dict[int(node)] = self.venuenode_embeddings.weight[idx].detach().cpu().clone()
            
        return paper_dict, venue_dict, loss.detach().item()


# # Example usage:

# # Assuming 'dm' is your data matrix
# trainer = NodeEmbeddingTrainer(dm, embedding_dim=2, num_epochs=10, lr=0.01, alpha=3)
# trainer.train()  # Train embeddings

# # Get the resulting dictionaries with embeddings
# paper_dict, venue_dict = trainer.get_embeddings()

# # Optionally: Print the dictionaries
# print("Paper Embeddings:", paper_dict)
# print("Venue Embeddings:", venue_dict)