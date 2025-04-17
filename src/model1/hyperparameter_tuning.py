import torch
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Packages.mini_batches import mini_batches_code
from Packages.embed_trainer import NodeEmbeddingTrainer
from Packages.data_divide import paper_c_paper_train
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt

# Number of iterations (adjust as needed)
# num_iterations =  622 # we need to be able to look at the complete dataset

# hyperparameters
batch_size = [3] # never (3) as it is seen as range when in a tuple
num_epochs = [40]
lr = [0.1]
alpha = [0.1,0.05]
lam = [0.001,0.01]

params = [{'name': 'batch_size', 'type': 'discrete', 'domain': batch_size}, 
          {'name': 'num_epochs', 'type': 'discrete', 'domain': num_epochs},
          {'name': 'lr', 'type': 'discrete', 'domain': lr},
          {'name': 'alpha', 'type': 'discrete', 'domain': alpha},
          {'name': 'lam', 'type': 'discrete', 'domain': lam}]

# # Modify the domain definitions to reflect ranges for continuous variables
# params = [
#     {'name': 'batch_size', 'type': 'discrete', 'domain': batch_size}, 
#     {'name': 'num_epochs', 'type': 'discrete', 'domain': num_epochs},
#     {'name': 'lr', 'type': 'continuous', 'domain': (0.001, 0.1)},  # Define a range for lr
#     {'name': 'alpha', 'type': 'continuous', 'domain': (0.01, 0.1)},  # Define a range for alpha
#     {'name': 'lam', 'type': 'continuous', 'domain': (0.001, 0.1)}    # Define a range for lam
# ]


def objective_function(params):
    # Load initial embeddings
    embed_venue = torch.load("dataset/ogbn_mag/processed/venue_embeddings.pt")
    embed_paper = torch.load("dataset/ogbn_mag/processed/paper_embeddings.pt")
    data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

    # Initialize dictionaries to store embeddings
    paper_dict = {k: v.clone() for k, v in embed_paper.items()}
    venue_dict = {k: v.clone() for k, v in embed_venue.items()}

    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes

    num_iterations =  1
    for i in range(num_iterations):
        print(f"Iteration {i+1}")
        print(f"Batch size: {params[0][0]}, Num epochs: {params[0][1]}, LR: {params[0][2]}, Alpha: {params[0][3]}, Lambda: {params[0][4]}")

        # Generate mini-batches
        mini_b = mini_batches_code(paper_c_paper_train, l_prev, int(params[0][0]), ('paper', 'cites', 'paper'),data)
        dm, l_next, remapped_datamatrix_tensor,random_sample = mini_b.node_mapping()

        # Train embeddings and update dictionaries **in place**
        N_emb = NodeEmbeddingTrainer(
            dm=dm,
            remapped_datamatrix_tensor=remapped_datamatrix_tensor,
            paper_dict=paper_dict,  # Pass reference (no copy)
            venue_dict=venue_dict,
            num_epochs=int(params[0][1]),
            lr=params[0][2],
            alpha=params[0][3],
            lam=params[0][4]
        )
        paper_dict, venue_dict, loss = N_emb.train()  # Directly update original dictionaries

        # Update node list for the next iteration
        l_prev = l_next
    return loss

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = params,         # box-constrains of the problem
                                              acquisition_type = 'MPI',      # Select acquisition function MPI, EI, LCB
                                             )

opt.acquisition.exploration_weight=.1

opt.run_optimization(max_iter = 15) 

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: batch_size =" + str(x_best[0]) + ", num_epochs =" + str(x_best[1]) + ", lr =" + str(
    x_best[2])  + ", alpha =" + str(
    x_best[3])   + ", lam =" + str(
    x_best[4]))

