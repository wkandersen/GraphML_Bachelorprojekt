import torch
import torch.nn.functional as F

class LossFunction:
    def __init__(self, alpha=1.0, lam=0.01, weight=1.0, venue_weight=15, neg_ratio=5, eps=1e-8, use_regularization=False):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.eps = eps
        self.use_regularization = use_regularization
        self.lam = lam
        self.weight = weight
        self.venue_weight = venue_weight
        self.neg_ratio = neg_ratio

    def edge_probability(self, z_i, z_j):
        """Compute the probability of an edge existing between two embeddings."""
        cos_sim = F.cosine_similarity(z_i, z_j, dim=1)
        return torch.sigmoid(cos_sim - self.alpha)

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

        # 2. Select all positive and sample some negatives
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_indices = pos_mask.nonzero(as_tuple=False).flatten()
        neg_indices = neg_mask.nonzero(as_tuple=False).flatten()


        num_pos = pos_indices.numel()
        num_neg_sample = min(neg_indices.numel(), num_pos * self.neg_ratio)

        sampled_neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=labels.device)[:num_neg_sample]]

        keep_indices = torch.cat([pos_indices, sampled_neg_indices])

        # 3. Filter data using selected indices
        labels = labels[keep_indices]
        u_idx = u_idx[keep_indices]
        v_idx = v_idx[keep_indices]
        pv1_idx = pv1_idx[keep_indices]     
        pv2_idx = pv2_idx[keep_indices]

        z_u_list = []
        z_v_list = []
        valid_labels = []

        edge_entities = {
            0: 'paper',
            1: 'author',
            2: 'institution',
            3: 'field_of_study',
            4: 'venue'
        }
        venue_mask_list = []

        for i, j, i2, j2, label in zip(u_idx, pv1_idx, v_idx, pv2_idx, labels):
            ent_type_u = edge_entities[j.item()]
            ent_type_v = edge_entities[j2.item()]
            id_u = i.item()
            id_v = i2.item()

            if id_u not in z[ent_type_u] or id_v not in z[ent_type_v]:
                continue  # skip this edge
            
            z_u_list.append(z[ent_type_u][id_u])
            z_v_list.append(z[ent_type_v][id_v])
            valid_labels.append(label)
            
            # Construct venue mask entry based on original j/j2
            is_venue = (j.item() == 4) or (j2.item() == 4)
            venue_mask_list.append(is_venue)


        # Final tensors
        if z_u_list is torch.empty or z_v_list is torch.empty or not valid_labels:
            return 
        z_u = torch.stack(z_u_list)
        z_v = torch.stack(z_v_list)
        labels = torch.tensor(valid_labels, device=z_u.device)


        venue_mask = torch.tensor(venue_mask_list, device=z_u.device, dtype=torch.bool)


        # Compute loss with venue-aware weighting
        link_loss = self.link_loss(labels, z_u, z_v, venue_mask)

        loss = -torch.mean(link_loss)

        if self.use_regularization:
            regularization = self.lam * (torch.sum(z_u ** 2) + torch.sum(z_v ** 2))
            loss += regularization

        return loss
        
    def get_probs_and_labels(self, z, datamatrix_tensor):
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

        z_u = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(u_idx, pv1_idx)
        ])
        z_v = torch.stack([
            z[edge_entities[j.item()]][i.item()]
            for i, j in zip(v_idx, pv2_idx)
        ])

        probs = self.edge_probability(z_u, z_v)
        probs = torch.clamp(probs, self.eps, 1 - self.eps)

        return probs.detach(), labels.detach()
    
    def get_probs_and_labels_hybrid(self, z, datamatrix_tensor, MLP):
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

        z_u_list = []
        z_v_list = []
        valid_labels = []

        for i, j, i2, j2, label in zip(u_idx, pv1_idx, v_idx, pv2_idx, labels):
            ent_type_u = edge_entities[j.item()]
            ent_type_v = edge_entities[j2.item()]
            id_u = i.item()
            id_v = i2.item()

            if id_u not in z[ent_type_u] or id_v not in z[ent_type_v]:
                continue

            # --- U NODE ---
            if ent_type_u == 'paper':
                z_u = z['paper'][id_u]
                mlp_u = MLP[id_u]
                z_u_done = torch.cat((z_u, mlp_u.detach()), dim=-1)
            elif ent_type_u == 'venue':
                z_u_done = z['venue'][id_u]
            else:
                raise ValueError(f"Unexpected ent_type_u: {ent_type_u}")

            # --- V NODE ---
            if ent_type_v == 'paper':
                z_v = z['paper'][id_v]
                mlp_v = MLP[id_v]
                z_v_done = torch.cat((z_v, mlp_v.detach()), dim=-1)
            elif ent_type_v == 'venue':
                z_v_done = z['venue'][id_v]
            else:
                raise ValueError(f"Unexpected ent_type_v: {ent_type_v}")

            z_u_list.append(z_u_done)
            z_v_list.append(z_v_done)
            valid_labels.append(label)

        if not z_u_list or not z_v_list or not valid_labels:
            return None, None  # or raise an error

        z_u = torch.stack(z_u_list)
        z_v = torch.stack(z_v_list)
        labels = torch.tensor(valid_labels, device=z_u.device)

        probs = self.edge_probability(z_u, z_v)
        probs = torch.clamp(probs, self.eps, 1 - self.eps)

        return probs.detach(), labels.detach()

    

    def compute_loss_hybrid(self, z, datamatrix_tensor, MLP):
        """Compute the total loss for the dataset."""
        labels = datamatrix_tensor[:, 0].float()
        u_idx = datamatrix_tensor[:, 1].long()
        v_idx = datamatrix_tensor[:, 2].long()
        pv1_idx = datamatrix_tensor[:, 3].long()
        pv2_idx = datamatrix_tensor[:, 4].long()

        # 2. Select all positive and sample some negatives
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_indices = pos_mask.nonzero(as_tuple=False).flatten()
        neg_indices = neg_mask.nonzero(as_tuple=False).flatten()


        num_pos = pos_indices.numel()
        num_neg_sample = min(neg_indices.numel(), num_pos * self.neg_ratio)

        sampled_neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=labels.device)[:num_neg_sample]]

        keep_indices = torch.cat([pos_indices, sampled_neg_indices])

        # 3. Filter data using selected indices
        labels = labels[keep_indices]
        u_idx = u_idx[keep_indices]
        v_idx = v_idx[keep_indices]
        pv1_idx = pv1_idx[keep_indices]     
        pv2_idx = pv2_idx[keep_indices]

        z_u_list = []
        z_v_list = []
        valid_labels = []

        edge_entities = {
            0: 'paper',
            1: 'author',
            2: 'institution',
            3: 'field_of_study',
            4: 'venue'
        }
        venue_mask_list = []

        for i, j, i2, j2, label in zip(u_idx, pv1_idx, v_idx, pv2_idx, labels):
            ent_type_u = edge_entities[j.item()]
            ent_type_v = edge_entities[j2.item()]
            id_u = i.item()
            id_v = i2.item()

            if id_u not in z[ent_type_u] or id_v not in z[ent_type_v]:
                continue

            # --- U NODE ---
            if ent_type_u == 'paper':
                z_u = z['paper'][id_u]        # [2]
                mlp_u = MLP[id_u]             # [2]
                z_u_done = torch.cat((z_u, mlp_u.detach()), dim=-1)  # [4]
            elif ent_type_u == 'venue':
                z_u_done = z['venue'][id_u]   # already [4]
            else:
                raise ValueError(f"Unexpected ent_type_u: {ent_type_u}")

            # --- V NODE ---
            if ent_type_v == 'paper':
                z_v = z['paper'][id_v]
                mlp_v = MLP[id_v]
                z_v_done = torch.cat((z_v, mlp_v.detach()), dim=-1)
            elif ent_type_v == 'venue':
                z_v_done = z['venue'][id_v]
            else:
                raise ValueError(f"Unexpected ent_type_v: {ent_type_v}")

            # Collect
            z_u_list.append(z_u_done)
            z_v_list.append(z_v_done)
            valid_labels.append(label)
            venue_mask_list.append(j.item() == 4 or j2.item() == 4)



        # Final tensors
        if z_u_list is torch.empty or z_v_list is torch.empty or not valid_labels:
            return 
        try:
            z_u = torch.stack(z_u_list)
            z_v = torch.stack(z_v_list)
        except RuntimeError as e:
            print("Error while stacking tensors!")
            print(f"z_u_list sizes: {[t.size() for t in z_u_list]}")
            print(f"z_v_list sizes: {[t.size() for t in z_v_list]}")
            raise e  # re-raise so you don't silently ignore the error

        labels = torch.tensor(valid_labels, device=z_u.device)


        venue_mask = torch.tensor(venue_mask_list, device=z_u.device, dtype=torch.bool)


        # Compute loss with venue-aware weighting
        link_loss = self.link_loss(labels, z_u, z_v, venue_mask)

        loss = -torch.mean(link_loss)

        if self.use_regularization:
            regularization = self.lam * (torch.sum(z_u ** 2) + torch.sum(z_v ** 2))
            loss += regularization

        return loss



    
    