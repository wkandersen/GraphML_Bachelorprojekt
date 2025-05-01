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

        # Get embeddings for u_idx and v_idx
        z_u = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(u_idx, pv1_idx)
        ])
        z_v = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(v_idx, pv2_idx)
        ])

        # Compute link loss for all edges in the batch
        link_loss = self.link_loss(labels, z_u, z_v)  # shape (B,)

        # Mean loss over the batch
        loss = -torch.mean(link_loss)

        # Optionally add regularization
        if self.use_regularization:
            regularization = self.lam * torch.sum(z ** 2)
            loss += regularization

        return loss
