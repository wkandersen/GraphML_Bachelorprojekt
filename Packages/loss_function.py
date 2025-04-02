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
            regularization = -self.lam * torch.sum(z ** 2)
            loss += regularization

        return loss



    # def edge_probability(self, z_i, z_j, type_i, type_j):
    #     """Compute the probability of an edge existing between two nodes, considering embeddings and types."""
    #     type_i = (type_i.view(1, -1).float())*0
    #     type_j = (type_j.view(1, -1).float())*0

    #     # Combine the node embeddings and types
    #     z_i = z_i.view(1, -1).float()  # Ensure z_i is a float tensor
    #     z_j = z_j.view(1, -1).float()  # Ensure z_j is a float tensor
        
    #     combined_i = torch.cat((z_i, type_i), dim=-1)  # Concatenate embedding and type for node i
    #     combined_j = torch.cat((z_j, type_j), dim=-1)  # Concatenate embedding and type for node j
        
    #     dist = torch.norm(combined_i - combined_j) ** 2  # Squared Euclidean distance
    #     return 1 / (1 + torch.exp(-self.alpha + dist))  # Logistic function

    # def link_loss(self, label, z_u, z_v, type_u, type_v):
    #     """Compute the loss for a single edge, considering node types."""
    #     prob = self.edge_probability(z_u, z_v, type_u, type_v)
    #     prob = torch.clamp(prob, self.eps, 1 - self.eps)  # Numerical stability

    #     return label.float() * torch.log(prob) + (1 - label.float()) * torch.log(1 - prob)

    # def compute_loss(self, z, types, datamatrix_tensor):
    #     """Compute the total loss for the dataset, considering node types."""
    #     sum_loss = sum(
    #         self.link_loss(label, z[u_idx], z[v_idx], types[u_idx][0], types[v_idx][1])
    #         for label, u_idx, v_idx in datamatrix_tensor)
        

    #     loss = -sum_loss / len(datamatrix_tensor)

    #     if self.use_regularization:
    #         regularization = self.lam * torch.sum(z ** 2)
    #         loss += regularization

    #     return loss
    

# loss_fn = LossFunction(alpha=1.0, use_regularization=True)
# loss_value = loss_fn.compute_loss(z, datamatrix_tensor)