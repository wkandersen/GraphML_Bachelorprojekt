import random
import itertools
import torch

class mini_batches_code:
    def __init__(self, data, unique_list, sample_size, edge_type, full_data):
        self.data = data
        self.sample_size = sample_size
        self.edge_type = edge_type
        self.unique_list = unique_list
        self.full_data = full_data
        self.device = self.data.device if isinstance(self.data, torch.Tensor) else torch.device("cpu")

    def get_batch(self):
        unique_list = self.unique_list  # Local reference to avoid repeated attribute access
        
        if len(unique_list) < self.sample_size:
            # Case 1: Not enough samples remaining
            random_sample = unique_list
            sample_tensor = torch.as_tensor(random_sample, device=self.device)  # More efficient than torch.tensor
            mask = torch.isin(self.data[0], sample_tensor)
            filtered_data = self.data[:, mask]
            return filtered_data, random_sample, []
        else:
            # Case 2: Normal sampling case
            # Optimized random sampling without modifying the original list
            random_sample = random.sample(unique_list, self.sample_size)
            
            # Create remaining list using set difference (faster than repeated remove())
            remaining_list = list(set(unique_list) - set(random_sample))
            
            # Vectorized operations for filtering
            sample_tensor = torch.as_tensor(random_sample, device=self.device)
            mask = torch.isin(self.data[0], sample_tensor)
            filtered_data = self.data[:, mask]
            
            return filtered_data, random_sample, remaining_list

    # def data_matrix(self):
    #     # Precompute constants and data
    #     data = self.full_data
    #     edge_entities = {
    #         'paper': 0,
    #         'author': 1,
    #         'institution': 2,
    #         'field_of_study': 3,
    #         'venue': 4,
    #     }
        
    #     # Get batch data
    #     tensor, random_sample, unique_list = self.get_batch()
        
    #     # Initialize result tensor
    #     if tensor.shape[1] == 0:
    #         result_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)
    #     else:
    #         # Vectorized creation of result tensor
    #         edge_type1 = edge_entities[self.edge_type[0]]
    #         edge_type2 = edge_entities[self.edge_type[2]]
    #         ones = torch.ones(tensor.shape[1], device=self.device)
    #         result_tensor = torch.stack([
    #             ones,
    #             tensor[0, :],
    #             tensor[1, :],
    #             torch.full((tensor.shape[1],), edge_type1, device=self.device),
    #             torch.full((tensor.shape[1],), edge_type2, device=self.device)
    #         ], dim=1).long()

    #     # Prepare for non-edges and venues
    #     non_edges = []
    #     venues = []
        
    #     # Precompute unique targets and paper venues
    #     unique_targets = tensor[1].unique() if tensor.shape[1] > 0 else torch.tensor([], device=self.device)
    #     paper_venues = data['y_dict']['paper']
        
    #     # Create a set of existing edges for faster lookup
    #     existing_edges = set()
    #     if tensor.shape[1] > 0:
    #         existing_edges = {(i.item(), j.item()) for i, j in zip(result_tensor[:, 1], result_tensor[:, 2])}
        
    #     # Generate non-edges and venues
    #     for i in random_sample:
    #         # Add venue edges
    #         venues.append(torch.tensor(
    #             [1, i, paper_venues[i], edge_entities[self.edge_type[0]], edge_entities['venue']],
    #             device=self.device
    #         ))
            
    #         # Add non-edges
    #         for j in unique_targets:
    #             if i != j and (i.item(), j.item()) not in existing_edges:
    #                 non_edges.append(torch.tensor(
    #                     [0, i, j.item(), edge_entities[self.edge_type[0]], edge_entities[self.edge_type[2]]],
    #                     device=self.device
    #                 ))
        
    #     # Generate venue non-edges using vectorized operations where possible
    #     for idx, (r, j) in enumerate(itertools.combinations(random_sample, 2)):
    #         r_venue = paper_venues[r]
    #         j_venue = paper_venues[j]
    #         if r_venue != j_venue:
    #             venues.append(torch.tensor(
    #                 [0, r, j_venue, edge_entities['paper'], edge_entities['venue']],
    #                 device=self.device
    #             ))
    #             venues.append(torch.tensor(
    #                 [0, j, r_venue, edge_entities['paper'], edge_entities['venue']],
    #                 device=self.device
    #             ))
        
    #     # Stack non-edges and venues only once
    #     non_edges_tensor = torch.stack(non_edges) if non_edges else torch.empty((0, 5), dtype=torch.long, device=self.device)
    #     venues_tensor = torch.stack(venues) if venues else torch.empty((0, 5), dtype=torch.long, device=self.device)
        
    #     # Concatenate all tensors
    #     data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor), dim=0)
        
    #     return data_matrix, unique_list, random_sample


    def data_matrix(self):
        data = self.full_data
        edge_entities = {
            'paper': 0,
            'author': 1,
            'institution': 2,
            'field_of_study': 3,
            'venue': 4,
        }

        tensor, random_sample, unique_list = self.get_batch()

        if tensor.shape[1] == 0:
            result_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)
        else:
            edge_type1 = edge_entities[self.edge_type[0]]
            edge_type2 = edge_entities[self.edge_type[2]]
            ones = torch.ones(tensor.shape[1], device=self.device)
            result_tensor = torch.stack([
                ones,
                tensor[0, :],
                tensor[1, :],
                torch.full((tensor.shape[1],), edge_type1, device=self.device),
                torch.full((tensor.shape[1],), edge_type2, device=self.device)
            ], dim=1).long()

        paper_venues = data['y_dict']['paper']
        edge_type1 = edge_entities[self.edge_type[0]]
        edge_type2 = edge_entities[self.edge_type[2]]

        random_sample_tensor = torch.tensor(random_sample, device=self.device)

        # VENUES: add positive venue edges
        venue_targets = paper_venues[random_sample_tensor]
        venues_tensor = torch.stack([
            torch.ones(len(random_sample), device=self.device),
            random_sample_tensor,
            venue_targets.flatten(),
            torch.full((len(random_sample),), edge_type1, device=self.device),
            torch.full((len(random_sample),), edge_entities['venue'], device=self.device)
        ], dim=1).long()

        # NON-EDGES: create all possible (i, j) pairs
        if tensor.shape[1] > 0:
            unique_targets = tensor[1].unique()
            i_vals = random_sample_tensor.repeat_interleave(len(unique_targets))
            j_vals = unique_targets.repeat(len(random_sample_tensor))

            existing_edges = result_tensor[:, 1:3]
            edge_pairs = torch.stack((i_vals, j_vals), dim=1)

            mask = ~((edge_pairs[:, None] == existing_edges).all(dim=2).any(dim=1)) & (i_vals != j_vals)
            non_edges_pairs = edge_pairs[mask]

            if non_edges_pairs.shape[0] > 0:
                non_edges_tensor = torch.cat([
                    torch.zeros((non_edges_pairs.shape[0], 1), device=self.device, dtype=torch.long),
                    non_edges_pairs,
                    torch.full((non_edges_pairs.shape[0], 1), edge_type1, device=self.device),
                    torch.full((non_edges_pairs.shape[0], 1), edge_type2, device=self.device)
                ], dim=1)
            else:
                non_edges_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)
        else:
            non_edges_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)

        # VENUE NON-EDGES: (r, j) where r_venue != j_venue
        comb_r, comb_j = torch.combinations(random_sample_tensor, r=2).unbind(1)
        r_venue = paper_venues[comb_r]
        j_venue = paper_venues[comb_j]
        
        unequal_mask = r_venue != j_venue
        unequal_mask = unequal_mask.flatten()
        if unequal_mask.any():
            comb_r = comb_r[unequal_mask]
            comb_j = comb_j[unequal_mask]
            r_venue = r_venue[unequal_mask]
            j_venue = j_venue[unequal_mask]

            comb_r = comb_r.squeeze()
            comb_j = comb_j.squeeze()
        
            venue_non_edges = torch.cat([
                torch.zeros((comb_r.shape[0]*2, 1), device=self.device, dtype=torch.long),
                torch.cat([comb_r.unsqueeze(1), comb_j.unsqueeze(1)], dim=0),
                torch.cat([j_venue, r_venue], dim=0),
                torch.full((comb_r.shape[0]*2, 1), edge_entities['paper'], device=self.device),
                torch.full((comb_r.shape[0]*2, 1), edge_entities['venue'], device=self.device)
            ], dim=1)
        else:
            venue_non_edges = torch.empty((0, 5), dtype=torch.long, device=self.device)

        # CONCAT everything
        data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor, venue_non_edges), dim=0)

        return data_matrix, unique_list, random_sample
