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
        if len(self.unique_list) < self.sample_size:
            # Put random_sample on the same device as self.data
            random_sample = self.unique_list
            sample_tensor = torch.tensor(random_sample, device=self.device)
            mask = torch.isin(self.data[0], sample_tensor)
            filtered_data = self.data[:, mask]
            return filtered_data, random_sample, []
        else:
            list_pcp = self.unique_list
            random_sample = random.sample(list_pcp, self.sample_size)
            print(random_sample)

            for value in random_sample:
                list_pcp.remove(value)

            # Put random_sample on the same device as self.data
            sample_tensor = torch.tensor(random_sample, device=self.device)
            mask = torch.isin(self.data[0], sample_tensor)
            filtered_data = self.data[:, mask]

            return filtered_data, random_sample, list_pcp

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
            result_tensor = torch.empty((0, 5), dtype=torch.long)
        else:
            result_tensor = torch.stack([
                torch.tensor([1, tensor[0, i], tensor[1, i], edge_entities[self.edge_type[0]], edge_entities[self.edge_type[2]]])
                for i in range(tensor.shape[1])
            ])

        non_edges, venues = [], []

        for i in random_sample:
            venues.append(torch.tensor(
                [1, i, data['y_dict']['paper'][i], edge_entities[self.edge_type[0]], edge_entities['venue']],
                device=self.device
            ))

            for j in tensor[1].unique():
                if i != j and not torch.any((result_tensor[:, 1] == i) & (result_tensor[:, 2] == j)):
                    non_edges.append(torch.tensor(
                        [0, i, j.item(), edge_entities[self.edge_type[0]], edge_entities[self.edge_type[2]]],
                        device=self.device
                    ))

        for r, j in itertools.combinations(random_sample, 2):
            if data['y_dict']['paper'][r] != data['y_dict']['paper'][j]:
                venues.append(torch.tensor([0, r, data['y_dict']['paper'][j], edge_entities['paper'], edge_entities['venue']], device=self.device))
                venues.append(torch.tensor([0, j, data['y_dict']['paper'][r], edge_entities['paper'], edge_entities['venue']], device=self.device))

        non_edges_tensor = torch.stack(non_edges) if non_edges else torch.empty((0, 5), dtype=torch.long, device=self.device)
        venues_tensor = torch.stack(venues) if venues else torch.empty((0, 5), dtype=torch.long, device=self.device)

        data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor), dim=0)
        return data_matrix, unique_list, random_sample

    def node_mapping(self):
        datamatrix_tensor, ul, random_sample = self.data_matrix()

        lm1 = torch.unique(torch.stack((datamatrix_tensor[:, 1], datamatrix_tensor[:, 3]), dim=1), dim=0)
        lm2 = torch.unique(torch.stack((datamatrix_tensor[:, 2], datamatrix_tensor[:, 4]), dim=1), dim=0)
        unique_global_node_ids = torch.unique(torch.cat([lm1, lm2], dim=0), dim=0)

        node_mapping = {
            (global_id.item(), type_id.item()): idx
            for idx, (global_id, type_id) in enumerate(unique_global_node_ids)
        }

        remapped_datamatrix_tensor = datamatrix_tensor.clone()

        remapped_datamatrix_tensor[:, 1] = torch.tensor([
            node_mapping[(global_id.item(), type_id.item())]
            for global_id, type_id in zip(datamatrix_tensor[:, 1], datamatrix_tensor[:, 3])
        ], device=self.device)

        remapped_datamatrix_tensor[:, 2] = torch.tensor([
            node_mapping[(global_id.item(), type_id.item())]
            for global_id, type_id in zip(datamatrix_tensor[:, 2], datamatrix_tensor[:, 4])
        ], device=self.device)

        return datamatrix_tensor, ul, remapped_datamatrix_tensor, random_sample
