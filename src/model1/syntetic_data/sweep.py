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
from Packages.mini_batches_fast import mini_batches_fast
from Packages.plot_embeddings import set_seed, plot_paper_venue_embeddings, plot_pos_neg_histograms
from venue_homogeneity import compute_venue_homogeneity
from pprint import pprint
from torch.nn import Parameter
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance

# Sweep configuration
sweep_configuration = {
    "method": "random",  # Can also use "bayes"
    "name": "syn_sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {"values": [10]},
        "num_epochs": {"values": [15]},
        "lr": {"values": [0.1,0.01]},
        "alpha": {"values": [0.01,1]},
        "lam": {"values": [0,0.001]},
        "weight": {"values": [0.01,0.1,1]},
        "venue_weight": {"values": [10,50,100]},
    }
}
trials = 15

def sweep_objective():
    run = wandb.init()
    config = wandb.config
    embedding_dim = 8
    b = 1
    save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b{b}.pt')
    paper_c_paper_train = save['paper_c_paper_train']
    data,venue_value = save['data'], save['venue_value']
    num_papers = len(paper_c_paper_train.unique())
    set_seed(69)
    text = 'test'

    embed_dict = save['collected_embeddings']

    loss_function = LossFunction(alpha=config.alpha,weight=config.weight,lam=config.lam,venue_weight=config.venue_weight)
    N_emb = NodeEmbeddingTrainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    citation_dict = defaultdict(list)
    for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
        citation_dict[src.item()].append(tgt.item())

    all_papers = list(citation_dict.keys())

    num_iterations = int(len(embed_dict['venue']) + len(embed_dict['paper'])) # we need to be able to look at the complete dataset

# num_iterations = 2

    # Before training
    with open("embedding_output_before.txt", "w") as f:
        pprint(embed_dict, stream=f, indent=2, width=80)

    # plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=embed_dict,sample_size=num_papers,filename=file_before)

    for group in embed_dict:
        for key in embed_dict[group]: 
            embed_dict[group][key] = Parameter(
                embed_dict[group][key].clone().detach().to(device),
                requires_grad=True
            )

    params = []
    for group in embed_dict.values():  # e.g., embed_dict['paper'], embed_dict['venue']
        for param in group.values():   # e.g., embed_dict['paper'][123]
            params.append(param)

    optimizer = torch.optim.Adam(params, lr=config.lr)

    loss_pr_epoch = []
    loss_ven_epoch = []
    loss_pap_epoch = []
    for i in range(config.num_epochs):
        print(f"Epoch {i + 1}/{config.num_epochs}")
        l_prev = list(paper_c_paper_train.unique().numpy())  # Initial list of nodes
        loss_pr_iteration = []
        loss_ven_iteration = []
        loss_pap_iteration = []
        all_pos_probs = []
        all_neg_probs = []

        mini_b = mini_batches_fast(paper_c_paper_train, l_prev, config.batch_size, ('paper', 'cites', 'paper'), data, citation_dict, all_papers)

        for j in range(num_iterations):
            mini_b.set_unique_list(l_prev)  # Update only the node list
            dm, l_next, random_sample = mini_b.data_matrix()
            dm_ven_pap = dm[dm[:,4] == 4]
            dm_pap_pap = dm[dm[:,4] != 4]

            # Move data to GPU
            dm = dm.to(device)
            optimizer.zero_grad()
            loss = loss_function.compute_loss(embed_dict, dm)
            loss_ven = loss_function.compute_loss(embed_dict,dm_ven_pap)
            loss_pap = loss_function.compute_loss(embed_dict,dm_pap_pap)
            loss.backward()
            optimizer.step()

            wandb.log({"loss_batch": loss.detach().item()})
            wandb.log({"paper_loss": loss_pap.detach().item()})
            wandb.log({"venue_loss": loss_ven.detach().item()})

                    # Get predicted probabilities and labels for this batch

            # print(f"Loss: {loss.detach().item()}")
            # Update node list for the next iteration
            loss_pr_iteration.append(loss.detach().item())
            loss_ven_iteration.append(loss_ven.detach().item())
            loss_pap_iteration.append(loss_pap.detach().item())

            l_prev = l_next


            if len(l_next) == 0:
                print("No more nodes to process. Exiting.")
                print(loss_pr_iteration)
                loss_pr_epoch.append(np.mean(loss_pr_iteration))
                loss_ven_epoch.append(np.mean(loss_ven_iteration))
                loss_pap_epoch.append(np.mean(loss_pap_iteration))
                print(f"loss_epoch: {loss_pr_epoch[i]}")
                wandb.log({"loss": loss_pr_epoch[i]})
                break
    
    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Bachelor_projekt")
    wandb.agent(sweep_id, function=sweep_objective, count=trials)  # Run 15 trials