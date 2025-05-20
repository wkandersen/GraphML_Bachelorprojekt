import torch
import copy
import sys
import os
import gc
import wandb
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from Packages.loss_function import LossFunction
from Packages.embed_trainer import NodeEmbeddingTrainer
from src.mini_batches_fast import mini_batches_fast
from Packages.create_syn_data import data,paper_c_paper_train, collected_embeddings

embed_dict = collected_embeddings

batch_size = 7
num_epochs = 1
lr = 0.1
alpha = 0.1
lam = 0.01
embedding_dim = 2

loss_function = LossFunction()
N_emb = NodeEmbeddingTrainer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("starting")


citation_dict = defaultdict(list)
for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
    citation_dict[src.item()].append(tgt.item())

all_papers = list(citation_dict.keys())

num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper'])) # we need to be able to look at the complete dataset

# num_iterations = 2


params = []
for subdict in embed_dict.values():
    params.extend(subdict.values())
loss_pr_epoch = []

for i in range(num_epochs):
    print(f"Epoch {i + 1}/{num_epochs}")
    l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_pr_iteration = []

    # import time
    # start = time.time()
    # dm, unique_list, random_sample = mini_b.data_matrix()
    # print("Batch gen time:", time.time() - start)

    mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data, citation_dict, all_papers)

    for j in range(num_iterations):
        mini_b.set_unique_list(l_prev)  # Update only the node list
        dm, l_next, random_sample = mini_b.data_matrix()
        # print(dm)

    # for j in range(num_iterations):

        # Generate mini-batches
        # mini_b = mini_batches_fast(paper_c_paper_train, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
        # dm, l_next, random_sample = mini_b.data_matrix()

        # Move data to GPU
        dm = dm.to(device)
        optimizer.zero_grad()
        loss = loss_function.compute_loss(embed_dict, dm)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.detach().item()}")
        # Update node list for the next iteration
        loss_pr_iteration.append(loss.detach().item())

        l_prev = l_next
        

        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            print(loss_pr_iteration)
            loss_pr_epoch.append(np.mean(loss_pr_iteration))
            print(f"loss_epoch: {loss_pr_epoch[i]}")
            break

        # Cleanup
        if (i + 1) % 5 == 0:  # Or do it every iteration if memory is super tight
            import gc
            gc.collect()
            torch.cuda.empty_cache()