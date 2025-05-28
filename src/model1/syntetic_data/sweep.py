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
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator


def embed_valid_funct(paper_dict):

    for group_key in paper_dict:  # 'paper', 'venue'
        for id_key in paper_dict[group_key]:
            paper_dict[group_key][id_key] = paper_dict[group_key][id_key].detach().clone().cpu()
        set_seed(45)

    def get_mean_median_embedding(paper_dict, ent_type='paper', method='mean'):
        # Extract all embeddings for the given entity type
        embeddings = list(paper_dict[ent_type].values())  # list of tensors

        if not embeddings:
            raise ValueError(f"No embeddings found for entity type: {ent_type}")

        # Stack into a single tensor of shape [N, D]
        stacked = torch.stack(embeddings, dim=0)

        if method == 'mean':
            return stacked.mean(dim=0)  # [D]
        elif method == 'median':
            return stacked.median(dim=0).values  # [D]
        else:
            raise ValueError(f"Unsupported method: {method}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.1
    num_epochs = 100
    alpha = 0.1
    eps = 0.001
    lam = 0.01
    batch_size = 1
    embedding_dim = 2

    save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b1.pt')
    paper_c_paper_train,paper_c_paper_valid = save['paper_c_paper_train'], save['paper_c_paper_valid']
    data,venue_value = save['data'], save['venue_value']

    num_iterations = len(paper_c_paper_valid[0])
    # Move all embeddings to device 
    for ent_type in paper_dict:
        for k, v in paper_dict[ent_type].items(): 
            paper_dict[ent_type][k] = v.to(device)

    mean_emb = get_mean_median_embedding(paper_dict=paper_dict)
    median_emb = get_mean_median_embedding(paper_dict=paper_dict,method='median')

    citation_dict = defaultdict(list)
    for src, tgt in zip(paper_c_paper_train[0], paper_c_paper_train[1]):
        citation_dict[src.item()].append(tgt.item())

    all_papers = list(citation_dict.keys())

    loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

    unique_train = set(paper_c_paper_train.flatten().unique().tolist())
    unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())
    valid_exclusive = unique_valid - unique_train
    l_prev = list(valid_exclusive)
    predictions = {}

    counter = 0
    acc = []
    for i in range(num_iterations):
        mini_b = mini_batches_fast(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'), data,citation_dict, all_papers,venues=True)
        dm, l_next, random_sample = mini_b.data_matrix()
        # print(random_sample)
        # print(paper_c_paper_valid)

        dm = dm[dm[:, 4] != 4]

        if len(dm) < 1:
            # Assign mean embedding to the sample
            paper_dict['paper'][random_sample[0]] = torch.nn.Parameter(median_emb.clone().to(device))

            # Directly evaluate using the mean embedding (skip training)
            true_label = int(venue_value[random_sample[0]].cpu().numpy())

            logi_f = []
            for j in range(len(paper_dict['venue'])):
                paper_emb = paper_dict['paper'][random_sample[0]].to(device)
                venue_emb = paper_dict['venue'][j].to(device)
                dist = torch.norm(paper_emb - venue_emb) ** 2
                logi = torch.sigmoid(alpha - dist)
                logi_f.append((logi.item(), j))

            logits, node_ids = zip(*logi_f)
            logi_f_tensor = torch.tensor(logits, device=device)
            softma = F.softmax(logi_f_tensor, dim=0)

            sorted_probs, sorted_indices = torch.sort(softma, descending=True)
            ranked_venue_ids = [node_ids[i] for i in sorted_indices.tolist()]
            true_class_rank = ranked_venue_ids.index(true_label) + 1 if true_label in ranked_venue_ids else -1

            predicted_node_id = ranked_venue_ids[0]
            highest_prob_value = sorted_probs[0].item()
            predictions[random_sample[0]] = (true_label, predicted_node_id, true_class_rank)

            items = sorted(predictions.items())
            y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
            y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

            evaluator = Evaluator(name='ogbn-mag')
            result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            acc.append(result['acc'])

            # l_prev = l_next
            continue
        # print(dm)

        test1 = dm[dm[:, 0] == 1]
        list_test = test1[:, 2].unique().tolist()

        embedding_list = []
        for test_item in list_test:
            if test_item in paper_dict['paper']:
                embedding_list.append(paper_dict['paper'][test_item])

        if len(embedding_list) == 0:
            new_embedding = torch.nn.Parameter(torch.randn(embedding_dim, device=device, requires_grad=True))
        else:
            mean_tensor = torch.mean(torch.stack(embedding_list), dim=0)
            new_embedding = torch.nn.Parameter(mean_tensor.clone().to(device))

            # print(f"Element-wise mean tensor: {mean_tensor}")



        paper_dict['paper'][random_sample[0]] = new_embedding
        new_optimizer = torch.optim.Adam([paper_dict['paper'][random_sample[0]]], lr=lr)


        prev_losses = []
        patience = 5
        tolerance = 1e-4  # Define how close "very close" means

        for epoch in range(num_epochs):
            new_optimizer.zero_grad()
            loss = loss_function.compute_loss(paper_dict, dm)
            if loss is None:
                print(f"[SKIP] Loss computation failed for sample {random_sample[0]}.")
                counter += 1
                break
            loss.backward()
            new_optimizer.step()
            
            current_loss = loss.detach().item()
            
            # Track loss history
            prev_losses.append(current_loss)

            if len(prev_losses) > patience:
                recent = prev_losses[-patience:]
                if max(recent) - min(recent) < tolerance:
                    print(f"[EARLY STOP] Loss converged after {epoch} epochs.")
                    break

        # print(paper_dict['paper'][random_sample[0]])
        true_label = int(venue_value[random_sample[0]].cpu().numpy())

        logi_f = []
        
        for j in range(len(paper_dict['venue'])):
            paper_emb = paper_dict['paper'][random_sample[0]].to(device)
            venue_emb = paper_dict['venue'][j].to(device)
            dist = torch.norm(paper_emb - venue_emb) ** 2
            logi = torch.sigmoid(alpha - dist)
            logi_f.append((logi.item(), j))

        logits, node_ids = zip(*logi_f)
        logi_f_tensor = torch.tensor(logits, device=device)
        softma = F.softmax(logi_f_tensor, dim=0)

        # Rank true label
        sorted_probs, sorted_indices = torch.sort(softma, descending=True)
        ranked_venue_ids = [node_ids[i] for i in sorted_indices.tolist()]
        if true_label in ranked_venue_ids:
            true_class_rank = ranked_venue_ids.index(true_label) + 1  # 1-based
        else:
            true_class_rank = -1  # Not found (shouldn't happen)

        # Store prediction and rank info
        predicted_node_id = ranked_venue_ids[0]
        highest_prob_value = sorted_probs[0].item()
        predictions[random_sample[0]] = (true_label, predicted_node_id, true_class_rank)

        l_prev = l_next
        new_embedding = new_embedding.detach()


    items = sorted(predictions.items())
    y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
    y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

    evaluator = Evaluator(name='ogbn-mag')
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
    return result['acc']



# Sweep configuration
sweep_configuration = {
    "method": "random",  # Can also use "bayes"
    "name": "syn_sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {"values": [32,64,128]},
        "lr": {"values": [0.01,0.1,0.5]},
        "alpha": {"values": [0.01,0.1,0.5,1]},
        "lam": {"values": [0,0.001,0.01]},
    }
}
trials = 3

def sweep_objective():
    run = wandb.init()


    config = wandb.config
    config.weight = 0.1
    config.venue_weight = 100
    config.num_epochs = 20

    embedding_dim = 2
    b = 1
    save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b{b}.pt')
    paper_c_paper_train = save['paper_c_paper_train']
    data,venue_value = save['data'], save['venue_value']
    num_papers = len(paper_c_paper_train.unique())
    set_seed(69)
    text = 'test'

    embed_dict = save['collected_embeddings']

    loss_function = LossFunction(alpha=config.alpha,weight=config.weight,lam=config.lam,venue_weight=config.venue_weight,use_regularization=True)
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
        

    acc = embed_valid_funct(embed_dict)
    wandb.log({"acc": acc})
    
    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Bachelor_projekt")
    wandb.agent(sweep_id, function=sweep_objective, count=trials)  # Run 15 trials

