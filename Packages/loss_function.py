# import torch

# class LossFunction:
#     def __init__(self, alpha=1.0, eps=1e-8, use_regularization=False, lam=0.01):
#         """
#         Initialize the loss function with given parameters.
        
#         Args:
#             alpha (float): Scaling parameter for edge probability.
#             eps (float): Small value to prevent log(0).
#             use_regularization (bool): Whether to include Gaussian regularization.
#         """
#         self.alpha = alpha
#         self.eps = eps
#         self.use_regularization = use_regularization
#         self.lam = lam

#     def edge_probability(self, z_i, z_j):
#         """Compute the probability of an edge existing between two embeddings."""
#         dist = torch.norm(z_i - z_j) ** 2  # Squared Euclidean distance
#         return 1 / (1 + torch.exp(-self.alpha + dist))  # Logistic function

    # def link_loss(self, label, z_u, z_v):
    #     """Compute the loss for a single edge."""
    #     prob = self.edge_probability(z_u, z_v)
    #     prob = torch.clamp(prob, self.eps, 1 - self.eps)  # Numerical stability

    #     return label.float() * torch.log(prob) + (1 - label.float()) * torch.log(1 - prob)        

    # def compute_loss(self, z, datamatrix_tensor):
    #     """Compute the total loss for the dataset."""
    #     sum_loss = sum(
    #         self.link_loss(label, z[u_idx], z[v_idx])
    #         for label, u_idx, v_idx in datamatrix_tensor
    #     )

    #     loss = -sum_loss / len(datamatrix_tensor)

    #     if self.use_regularization:
    #         regularization = -self.lam * torch.sum(z ** 2)
    #         loss += regularization

    #     return loss

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

        # Get embeddings for u_idx and v_idx
        z_u = z[u_idx]  # shape (B, D)
        z_v = z[v_idx]  # shape (B, D)

        # Compute link loss for all edges in the batch
        link_loss = self.link_loss(labels, z_u, z_v)  # shape (B,)

        # Mean loss over the batch
        loss = -torch.mean(link_loss)

        # Optionally add regularization
        if self.use_regularization:
            regularization = self.lam * torch.sum(z ** 2)
            loss += regularization

        return loss
