import torch
import sys
import os
import gc
from Packages.mini_batches_fast import mini_batches_fast
from Packages.loss_function import LossFunction
from Packages.plot_embeddings import set_seed, plot_paper_venue_embeddings
# from Packages.synthetic_data_mixture import paper_c_paper_train, paper_c_paper_valid, data,venues_values
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator
import traceback
from collections import defaultdict

set_seed(45)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
embedding_dim = 8

save = torch.load(f'src/model1/syntetic_data/embed_dict/save_dim{embedding_dim}_b1.pt')
paper_c_paper_train,paper_c_paper_valid = save['paper_c_paper_train'], save['paper_c_paper_valid']
data,venue_value = save['data'], save['venue_value']
print(venue_value)

num_iterations = len(paper_c_paper_valid[0])


check = torch.load(f'src/model1/syntetic_data/embed_dict/embed_dict_test.pt', map_location=device)
paper_dict = check

# Move all embeddings to device 
for ent_type in paper_dict:
    for k, v in paper_dict[ent_type].items(): 
        paper_dict[ent_type][k] = v.to(device)

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
for i in range(num_iterations):
    mini_b = mini_batches_fast(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'), data,citation_dict, all_papers,venues=False)
    dm, l_next, random_sample = mini_b.data_matrix()
    # print(random_sample)
    # print(paper_c_paper_valid)
    if len(dm) < 1:
        print(f"[SKIP] Too few nodes for venue comparison: {random_sample}")
        counter += 1
        continue

    dm = dm[dm[:, 4] != 4]
    # print(dm)

    test1 = dm[dm[:, 0] == 1]
    list_test = test1[:, 2].unique().tolist()

    # mini_btrain1 = mini_batches_fast(paper_c_paper_train, dm[:, 2].tolist(), len(dm[:, 2]), ('paper', 'cites', 'paper'), data,citation_dict, all_papers)
    # dmtrain1, ultrain1, random_sampletrain1 = mini_btrain1.data_matrix()

    # test1 = dmtrain1[dmtrain1[:, 4] != 4]
    # test2 = test1[test1[:, 0] == 1]
    # list_test = test2[:, 2].unique().tolist()

    # concat_dm = torch.cat((dm, dmtrain1), 0)
    # rows = [[0, random_sample[0], test_item, 0, 0] for test_item in list_test]
    # tensor_result = torch.tensor(rows).to(device)
    # concat_dm = concat_dm.to(device)
    # tensor_result = tensor_result.to(device)
    # final = torch.cat((concat_dm, tensor_result), 0).to(device)

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
    # print(paper_dict['paper'][random_sample[0]])
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
                print(f"[EARLY STOP] Loss converged after {epoch + 1} epochs.")
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

    items = sorted(predictions.items())
    y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
    y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

    evaluator = Evaluator(name='ogbn-mag')
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

    l_prev = l_next
    new_embedding = new_embedding.detach()
    del new_optimizer

    if (i + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

save_dir = f'dataset/ogbn_mag/processed/Predictions'
os.makedirs(save_dir, exist_ok=True)
torch.save(predictions, os.path.join(save_dir, f'pred_dict_{embedding_dim}.pt'))

# Filter paper_dict to only include papers for which predictions were made
filtered_paper_dict = {
    'paper': {k: v for k, v in paper_dict['paper'].items() if k in predictions},
    'venue': paper_dict['venue']
}
num_papers = len(filtered_paper_dict['paper'])

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=filtered_paper_dict,sample_size=num_papers,plot_params={
        "alpha": alpha,
        "lambda": lam,
        "lr": lr,
        "epochs": num_epochs,
        "dim": embedding_dim
    })

print(counter)