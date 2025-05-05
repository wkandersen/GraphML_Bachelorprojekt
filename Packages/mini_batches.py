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

    def data_matrix(self):
        # Precompute constants and data
        data = self.full_data
        edge_entities = {
            'paper': 0,
            'author': 1,
            'institution': 2,
            'field_of_study': 3,
            'venue': 4,
        }
        
        # Get batch data
        tensor, random_sample, unique_list = self.get_batch()
        
        # Initialize result tensor
        if tensor.shape[1] == 0:
            result_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)
        else:
            # Vectorized creation of result tensor
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

        # Prepare for non-edges and venues
        non_edges = []
        venues = []
        
        # Precompute unique targets and paper venues
        unique_targets = tensor[1].unique() if tensor.shape[1] > 0 else torch.tensor([], device=self.device)
        paper_venues = data['y_dict']['paper']
        
        # Create a set of existing edges for faster lookup
        existing_edges = set()
        if tensor.shape[1] > 0:
            existing_edges = {(i.item(), j.item()) for i, j in zip(result_tensor[:, 1], result_tensor[:, 2])}
        
        # Generate non-edges and venues
        for i in random_sample:
            # Add venue edges
            venues.append(torch.tensor(
                [1, i, paper_venues[i], edge_entities[self.edge_type[0]], edge_entities['venue']],
                device=self.device
            ))
            
            # Add non-edges
            for j in unique_targets:
                if i != j and (i.item(), j.item()) not in existing_edges:
                    non_edges.append(torch.tensor(
                        [0, i, j.item(), edge_entities[self.edge_type[0]], edge_entities[self.edge_type[2]]],
                        device=self.device
                    ))
        
        # Generate venue non-edges using vectorized operations where possible
        for idx, (r, j) in enumerate(itertools.combinations(random_sample, 2)):
            r_venue = paper_venues[r]
            j_venue = paper_venues[j]
            if r_venue != j_venue:
                venues.append(torch.tensor(
                    [0, r, j_venue, edge_entities['paper'], edge_entities['venue']],
                    device=self.device
                ))
                venues.append(torch.tensor(
                    [0, j, r_venue, edge_entities['paper'], edge_entities['venue']],
                    device=self.device
                ))
        
        # Stack non-edges and venues only once
        non_edges_tensor = torch.stack(non_edges) if non_edges else torch.empty((0, 5), dtype=torch.long, device=self.device)
        venues_tensor = torch.stack(venues) if venues else torch.empty((0, 5), dtype=torch.long, device=self.device)
        
        # Concatenate all tensors
        data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor), dim=0)
        
        return data_matrix, unique_list, random_sample
