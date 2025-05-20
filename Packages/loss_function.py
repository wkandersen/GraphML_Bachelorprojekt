import torch

class LossFunction:
    def __init__(self, alpha=1.0, eps=1e-8, use_regularization=False, lam=0.01, weight=1.0, venue_weight=1.5):
        """
        Initialize the loss function with given parameters.
        
        Args:
            alpha (float): Scaling parameter for edge probability.
            eps (float): Small value to prevent log(0).
            use_regularization (bool): Whether to include Gaussian regularization.
            lam (float): Regularization strength.
            weight (float): Weight for negative samples.
            venue_weight (float): Additional weight for edges involving a 'venue' entity.
        """
        self.alpha = alpha
        self.eps = eps
        self.use_regularization = use_regularization
        self.lam = lam
        self.weight = weight
        self.venue_weight = venue_weight

    def edge_probability(self, z_i, z_j):
        """Compute the probability of an edge existing between two embeddings."""
        dist_sq = torch.sum((z_i - z_j) ** 2, dim=1)
        return torch.sigmoid(-dist_sq + self.alpha)

    def link_loss(self, label, z_u, z_v, venue_mask):
        """Compute the loss for a batch of edges, applying special weighting for venue edges."""
        prob = self.edge_probability(z_u, z_v)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)

        base_loss = label * torch.log(prob) + self.weight * (1 - label) * torch.log(1 - prob)

        # Apply venue_weight where venue is involved
        weighted_loss = torch.where(venue_mask, self.venue_weight * base_loss, base_loss)

        return weighted_loss

    def compute_loss(self, z, datamatrix_tensor):
        """Compute the total loss for the dataset."""
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

        # Get embeddings
        z_u = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(u_idx, pv1_idx)
        ])
        z_v = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(v_idx, pv2_idx)
        ])

        # Create mask for edges involving venue
        venue_mask = (pv1_idx == 4) | (pv2_idx == 4)

        # Compute loss with venue-aware weighting
        link_loss = self.link_loss(labels, z_u, z_v, venue_mask)

        loss = -torch.mean(link_loss)

        if self.use_regularization:
            regularization = self.lam * (torch.sum(z_u ** 2) + torch.sum(z_v ** 2))
            loss += regularization

        return loss
