import random
import itertools
import torch

# from memory_profiler import profile

# @profile

class mini_batches_code:
    def __init__(self,data, unique_list, sample_size,edge_type,full_data):
        self.data = data
        self.sample_size = sample_size
        self.edge_type = edge_type
        self.unique_list = unique_list
        self.full_data = full_data

    # @profile
    def get_batch(self):
        # random.seed(99) 
        # torch.manual_seed(99)
        list_pcp = self.unique_list
        random_sample = random.sample(list_pcp, self.sample_size)
        print(random_sample)
        for value in random_sample:
            list_pcp.remove(value)
        mask = torch.isin(self.data[0], torch.tensor(random_sample))
        filtered_data = self.data[:,mask]
        return filtered_data, random_sample, list_pcp
    
    # @profile
    def data_matrix(self):
        data = self.full_data
        edge_entities = {
            'paper': 0,
            'author': 1,
            'institution': 2,
            'field_of_study': 3,
            'venue': 4,
        }
        # Get batch and initialize tensors
        tensor, random_sample, unique_list = self.get_batch()

        # Create result tensor from input batch
        result_tensor = torch.stack([torch.tensor([1, tensor[0, i], tensor[1, i],edge_entities[self.edge_type[0]],edge_entities[self.edge_type[2]]]) for i in range(tensor.shape[1])])

        # Initialize lists for non_edges and venues
        non_edges, venues = [], []

        # Add venue links for sampled nodes
        for i in random_sample:
            venues.append(torch.tensor([1, i.item(), data['y_dict']['paper'][i], edge_entities[self.edge_type[0]],edge_entities['venue']]))

            # Find non-existing edges
            for j in tensor[1].unique():
                if i != j and not torch.any((result_tensor[:, 1] == i) & (result_tensor[:, 2] == j)): 
                    non_edges.append(torch.tensor([0, i.item(), j.item(),edge_entities[self.edge_type[0]],edge_entities[self.edge_type[2]]]))

        for r, j in itertools.combinations(random_sample, 2):  # itertools generates all unique pairs
            if data['y_dict']['paper'][r] != data['y_dict']['paper'][j]:
                venues.append(torch.tensor([0, r, data['y_dict']['paper'][j],edge_entities['paper'],edge_entities['venue']]))
                venues.append(torch.tensor([0, j, data['y_dict']['paper'][r],edge_entities['paper'],edge_entities['venue']]))

        # Convert lists to tensors only once to optimize memory usage
        non_edges_tensor = torch.stack(non_edges) if non_edges else torch.empty((0, 5), dtype=torch.long)
        venues_tensor = torch.stack(venues) if venues else torch.empty((0, 5), dtype=torch.long)

        # Merge all tensors
        data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor), dim=0)
        return data_matrix, unique_list, random_sample
    
    # @profile
    def node_mapping(self):

        datamatrix_tensor,ul,random_sample = self.data_matrix()

        lm1 = torch.unique(torch.stack((datamatrix_tensor[:, 1], datamatrix_tensor[:, 3]), dim=1), dim=0)
        lm2 = torch.unique(torch.stack((datamatrix_tensor[:, 2], datamatrix_tensor[:, 4]), dim=1), dim=0)

        unique_global_node_ids = torch.unique(torch.cat([lm1, lm2], dim=0), dim=0)

        # Step 2: Create a mapping from global node IDs to local node indices
        node_mapping = {(global_id.item(), type_id.item()): idx 
                            for idx, (global_id, type_id) in enumerate(unique_global_node_ids)}

        # Step 3: Remap the indices in the datamatrix_tensor using the node_mapping
        # We are remapping columns 1 and 2 in the datamatrix (i.e., the source and destination node indices)
        remapped_datamatrix_tensor = datamatrix_tensor.clone()  # Clone the tensor to avoid modifying the original
        # Extract the global_id and type_id for remapping
        remapped_datamatrix_tensor[:, 1] = torch.tensor([
            node_mapping[(global_id.item(), type_id.item())]  
            for global_id, type_id in zip(datamatrix_tensor[:, 1], datamatrix_tensor[:, 3])  # Use both columns
        ])

        remapped_datamatrix_tensor[:, 2] = torch.tensor([
            node_mapping[(global_id.item(), type_id.item())]  
            for global_id, type_id in zip(datamatrix_tensor[:, 2], datamatrix_tensor[:, 4])  # Use both columns
        ])

        return datamatrix_tensor, ul, remapped_datamatrix_tensor, random_sample

# mini_b = mini_batches_code(paper_c_paper_train, list(paper_c_paper.unique().numpy()), 10,('paper', 'cites', 'paper'))
# dm,l1 = mini_b.data_matrix()
# mini_b1 = mini_batches_code(paper_c_paper_train, l1, 10,('paper', 'cites', 'paper'))