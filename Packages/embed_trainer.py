import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.loss_function import LossFunction
import wandb



class NodeEmbeddingTrainer:
    def __init__(self, device=None):        # Initialize input data, parameters, and setup
        self.device = device or torch.device("cpu")

        # Optimizers
        # self.optimizer = torch.optim.Adam([], lr=self.lr) # KOM TILBAGE

        # Loss function (assumed to be defined elsewhere)
        # self.loss_function = LossFunction(alpha=self.alpha, eps=self.eps, use_regularization=True, lam=self.lam)

    def save_checkpoint(self, path):
        checkpoint = {
            'collected_embeddings': self.collected_embeddings.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path, *args, **kwargs):
        obj = NodeEmbeddingTrainer(*args, **kwargs)
        checkpoint = torch.load(path)
        obj.papernode_embeddings.load_state_dict(checkpoint['papernode_embeddings'])
        obj.venuenode_embeddings.load_state_dict(checkpoint['venuenode_embeddings'])
        obj.optimizer.load_state_dict(checkpoint['optimizer'])
        obj.venue_optimizer.load_state_dict(checkpoint['venue_optimizer'])
        obj.specific_papernode_indices = checkpoint['specific_papernode_indices']
        obj.specific_venuenode_indices = checkpoint['specific_venuenode_indices']
        return obj



# # Example usage:

# # Assuming 'dm' is your data matrix
# trainer = NodeEmbeddingTrainer(dm, embedding_dim=2, num_epochs=10, lr=0.01, alpha=3)
# trainer.train()  # Train embeddings

# # Get the resulting dictionaries with embeddings
# paper_dict, venue_dict = trainer.get_embeddings()

# # Optionally: Print the dictionaries
# print("Paper Embeddings:", paper_dict)
# print("Venue Embeddings:", venue_dict)