import random
import itertools
import torch

class mini_batches_code:
    def __init__(self,data, unique_list, sample_size,edge_type):
        self.data = data
        self.sample_size = sample_size
        self.edge_type = edge_type
        self.unique_list = unique_list

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
    
    def data_matrix(self):
        data, _ = torch.load(r"GraphML_Bachelorprojekt/dataset/ogbn_mag/processed/geometric_data_processed.pt")
        edge_entities = {
            'paper': torch.tensor([1,0,0,0,0]),
            'author': torch.tensor([0,1,0,0,0]),
            'institution': torch.tensor([0,0,1,0,0]),
            'field_of_study': torch.tensor([0,0,0,1,0]),
            'venue': torch.tensor([0,0,0,0,1])
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
                venues.append(torch.tensor([0, r, data['y_dict']['paper'][j],1,0]))
                venues.append(torch.tensor([0, j, data['y_dict']['paper'][r],1,0]))

        # Convert lists to tensors only once to optimize memory usage
        non_edges_tensor = torch.stack(non_edges) if non_edges else torch.empty((0, 5), dtype=torch.long)
        venues_tensor = torch.stack(venues) if venues else torch.empty((0, 5), dtype=torch.long)

        # Merge all tensors
        data_matrix = torch.cat((result_tensor, non_edges_tensor, venues_tensor), dim=0)
        return data_matrix, unique_list


# mini_b = mini_batches_code(paper_c_paper_train, list(paper_c_paper.unique().numpy()), 10,('paper', 'cites', 'paper'))
# dm,l1 = mini_b.data_matrix()
# mini_b1 = mini_batches_code(paper_c_paper_train, l1, 10,('paper', 'cites', 'paper'))