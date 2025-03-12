import torch

class LossFunction:
    def __init__(self, alpha=1.0, eps=1e-8, use_regularization=False):
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

    def edge_probability(self, z_i, z_j):
        """Compute the probability of an edge existing between two embeddings."""
        dist = torch.norm(z_i - z_j) ** 2  # Squared Euclidean distance
        return 1 / (1 + torch.exp(-self.alpha + dist))  # Logistic function

    def link_loss(self, label, z_u, z_v):
        """Compute the loss for a single edge."""
        prob = self.edge_probability(z_u, z_v)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)  # Numerical stability

        return label.float() * torch.log(prob) + (1 - label.float()) * torch.log(1 - prob)

    def compute_loss(self, z, datamatrix_tensor):
        """Compute the total loss for the dataset."""
        sum_loss = sum(
            self.link_loss(label, z[u_idx], z[v_idx])
            for label, u_idx, v_idx in datamatrix_tensor
        )

        loss = -sum_loss / len(datamatrix_tensor)

        if self.use_regularization:
            regularization = -0.5 * torch.sum(z ** 2)
            loss += regularization

        return loss
    

# loss_fn = LossFunction(alpha=1.0, use_regularization=True)
# loss_value = loss_fn.compute_loss(z, datamatrix_tensor)