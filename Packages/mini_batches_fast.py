import torch
import random   
from line_profiler import profile
from Packages.mini_batches import mini_batches_code
import numpy as np
from collections import defaultdict


class mini_batches_fast(mini_batches_code):
    def __init__(self, data, unique_list, sample_size, edge_type, full_data, citation_dict, all_papers,venues=True):
        self.data = data
        self.sample_size = sample_size
        self.edge_type = edge_type
        self.unique_list = unique_list
        self.full_data = full_data
        self.device = self.data.device if isinstance(self.data, torch.Tensor) else torch.device("cpu")
        self.citation_dict = citation_dict
        self.all_papers = all_papers
        self.remaining_papers = set(all_papers)
        self.set_unique_list(unique_list) 
        self.venues = venues

    
    def set_unique_list(self, unique_list):
        if isinstance(unique_list, torch.Tensor):
            self.unique_tensor = unique_list.to(self.device)
        else:
            self.unique_tensor = torch.tensor(unique_list, dtype=torch.long, device=self.device)

    @profile
    def get_batch(self):
        unique_tensor = self.unique_tensor

        if len(unique_tensor) < self.sample_size:
            sample_tensor = unique_tensor
            sample_tensor_sorted, _ = sample_tensor.sort()
            idx = torch.searchsorted(sample_tensor_sorted, self.data[0])
            idx = idx.clamp(max=sample_tensor_sorted.size(0) - 1)
            mask = sample_tensor_sorted[idx] == self.data[0]
            filtered_data = self.data[:, mask]
            return filtered_data, sample_tensor.tolist(), []

        rand_idx = torch.randperm(len(unique_tensor), device=self.device)[:self.sample_size]
        sample_tensor = unique_tensor[rand_idx]
        sample_tensor_sorted, _ = sample_tensor.sort()

        idx = torch.searchsorted(sample_tensor_sorted, unique_tensor)
        idx = idx.clamp(max=sample_tensor_sorted.size(0) - 1)
        isin_mask = sample_tensor_sorted[idx] == unique_tensor
        remaining_tensor = unique_tensor[~isin_mask]

        idx = torch.searchsorted(sample_tensor_sorted, self.data[0])
        idx = idx.clamp(max=sample_tensor_sorted.size(0) - 1)
        mask = sample_tensor_sorted[idx] == self.data[0]
        filtered_data = self.data[:, mask]

        return filtered_data, sample_tensor.tolist(), remaining_tensor
    @profile
    def data_matrix(self):
        data = self.full_data
        edge_entities = {
            'paper': 0,
            'author': 1,
            'institution': 2,
            'field_of_study': 3,
            'venue': 4,
        }

        tensor, random_sample, unique_tensor = self.get_batch()

        edge_type1 = edge_entities[self.edge_type[0]]
        edge_type2 = edge_entities[self.edge_type[2]]

        if tensor.shape[1] == 0:
            result_tensor = torch.empty((0, 5), dtype=torch.long, device=self.device)
        else:
            ones = torch.ones(tensor.shape[1], device=self.device, dtype=torch.long)
            result_tensor = torch.stack([
                ones,
                tensor[0, :],
                tensor[1, :],
                torch.full((tensor.shape[1],), edge_type1, device=self.device),
                torch.full((tensor.shape[1],), edge_type2, device=self.device)
            ], dim=1).long()

        paper_venues = data['y_dict']['paper']
        random_sample_tensor = torch.tensor(random_sample, device=self.device).long()
        venue_targets = paper_venues[random_sample_tensor]

        if self.venues==True:
            venues_tensor = torch.stack([
                torch.ones(len(random_sample), device=self.device, dtype=torch.long),
                random_sample_tensor,
                venue_targets.flatten(),
                torch.full((len(random_sample),), edge_type1, device=self.device),
                torch.full((len(random_sample),), edge_entities['venue'], device=self.device)
            ], dim=1).long()

            
        comb_r, comb_j = torch.combinations(random_sample_tensor, r=2).unbind(1)
        r_venue = paper_venues[comb_r]
        j_venue = paper_venues[comb_j]
        unequal_mask = (r_venue != j_venue).flatten().nonzero(as_tuple=True)[0]

        if unequal_mask.numel() > 0:
            comb_r = comb_r[unequal_mask].flatten()
            comb_j = comb_j[unequal_mask].flatten()
            r_venue = r_venue[unequal_mask]
            j_venue = j_venue[unequal_mask]

            venue_non_edges = torch.cat([
                torch.zeros((comb_r.shape[0]*2, 1), device=self.device, dtype=torch.long),
                torch.cat([comb_r.unsqueeze(1), comb_j.unsqueeze(1)], dim=0),
                torch.cat([j_venue, r_venue], dim=0),
                torch.full((comb_r.shape[0]*2, 1), edge_entities['paper'], device=self.device),
                torch.full((comb_r.shape[0]*2, 1), edge_entities['venue'], device=self.device)
            ], dim=1)
        else:
            venue_non_edges = torch.empty((0, 5), dtype=torch.long, device=self.device)

        if tensor.shape[1] > 0:
            unique_targets = tensor[1].unique()
            i_grid, j_grid = torch.meshgrid(random_sample_tensor, unique_targets, indexing='ij')
            i_vals = i_grid.flatten()
            j_vals = j_grid.flatten()

            existing_edges = result_tensor[:, 1:3]
            max_node_id = max(i_vals.max().item(), j_vals.max().item(), existing_edges.max().item()) + 1

            packed_existing = (existing_edges[:, 0] * max_node_id + existing_edges[:, 1])
            packed_pairs = i_vals * max_node_id + j_vals

            exists = torch.isin(packed_pairs, packed_existing)
            mask = ~exists & (i_vals != j_vals)
            non_edges_pairs = torch.stack((i_vals[mask], j_vals[mask]), dim=1)

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

        if self.venues == True:
            data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor, venue_non_edges), dim=0)
        else:
            data_matrix = torch.cat((result_tensor, non_edges_tensor), dim=0)
        return data_matrix, unique_tensor, random_sample