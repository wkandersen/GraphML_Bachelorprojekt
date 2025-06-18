import torch
import sys
import os
import gc
from Packages.mini_batches_fast import mini_batches_fast
from Packages.loss_function_manhattan import LossFunction
from Packages.plot_embeddings import set_seed, plot_paper_venue_embeddings
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid, data,venue_value
# from Packages.synthetic_data_mixture import paper_c_paper_train, paper_c_paper_valid, data,venues_values
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb

run_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ckpt_dir = os.path.join("checkpoint/test", run_start_time)
os.makedirs(ckpt_dir, exist_ok=True)

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

def save_checkpoint(paper_dict, predictions, iteration, save_path, l_next, optimizer=None):
    checkpoint = {
        'collected_embeddings': paper_dict,
        'predictions': predictions,
        'iteration': iteration,
        'l_next': l_next,
        'optimizer': optimizer.state_dict() if optimizer else None
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at: {save_path}")

def evaluate_sample(paper_emb, venue_embeddings, venue_ids, true_label):
    dist_sq = torch.norm(venue_embeddings - paper_emb, dim=1, p=1)
    logits = torch.sigmoid(alpha - dist_sq)
    sorted_probs, sorted_indices = torch.sort(logits, descending=True)
    print(sorted_indices)
    ranked_venue_ids = [venue_ids[i] for i in sorted_indices.tolist()]
    predicted_node_id = ranked_venue_ids[0]
    true_class_rank = ranked_venue_ids.index(true_label) + 1 if true_label in ranked_venue_ids else -1
    return predicted_node_id, true_class_rank

saved_checkpoints = []
max_saved = 2

set_seed(45)

lr = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
embedding_dim = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
check = torch.load(f'checkpoint/2025-06-14_15-43-22/checkpoint_iter_64_{embedding_dim}_50_epoch_14_weight_0.1_with_optimizer.pt')
paper_dict = check['collected_embeddings']
# venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device)

print('past venue value error')
wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")
run = wandb.init(
    project="Forsvar",
    name=f"test_manhattan_{embedding_dim}_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "alpha": alpha,
        "lam": lam,
        "emb_dim": embedding_dim
    },
)

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

venue_ids = list(paper_dict['venue'].keys())
venue_emb_tensor = torch.stack([paper_dict['venue'][vid] for vid in venue_ids]).to(device)

for i in range(num_iterations):
    mini_b = mini_batches_fast(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'), data,citation_dict, all_papers,venues=True)
    dm, l_next, random_sample = mini_b.data_matrix()
    # print(random_sample)
    # print(paper_c_paper_valid)

    dm = dm[dm[:, 4] != 4]
    
    if len(dm) < 1:
        # Assign mean embedding to the sample
        paper_dict['paper'][random_sample[0]] = torch.nn.Parameter(mean_emb.clone().to(device))
        # Directly evaluate using the mean embedding (skip training)
        true_label = int(venue_value[random_sample[0]].cpu().numpy())
        pred_id, rank = evaluate_sample(
            paper_dict['paper'][random_sample[0]].to(device),
            venue_emb_tensor,
            venue_ids,
            true_label
        )
        predictions[random_sample[0]] = (true_label, pred_id, rank)
        l_prev = l_next
    else:
    # print(dm)
        relevant_papers = dm[dm[:, 0] == 1][:, 2].unique().tolist()
        embedding_list = [paper_dict['paper'][p] for p in relevant_papers if p in paper_dict['paper']]
        emb = torch.stack(embedding_list).mean(dim=0) if embedding_list else mean_emb.clone()
        new_embedding = torch.nn.Parameter(emb.clone().to(device))
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
                break  # exit epoch loop
            else:
                final_loss = loss.detach().item()
                loss.backward()
                new_optimizer.step()
            
            wandb.log({"loss": final_loss})
            
            # Track loss history
            prev_losses.append(final_loss)

            if len(prev_losses) > patience:
                recent = prev_losses[-patience:]
                if max(recent) - min(recent) < tolerance:
                    print(f"[EARLY STOP] Loss converged after {epoch} epochs.")
                    break
        
        if loss is not None:
            wandb.log({"final_loss": final_loss})
        else:
            continue

        # print(paper_dict['paper'][random_sample[0]])
        true_label = int(venue_value[random_sample[0]].cpu().numpy().item())

        # Evaluate the new embedding
        pred_id, rank = evaluate_sample(
            paper_dict['paper'][random_sample[0]].to(device),
            venue_emb_tensor,
            venue_ids,
            true_label
        )
        predictions[random_sample[0]] = (true_label, pred_id, rank)

        if len(l_next) == 0:
            print("No more nodes to process. Exiting.")
            break

        l_prev = l_next
        new_embedding = new_embedding.detach()

    # Save checkpoint every 10 iterations
    if i % 100 == 0 or i == num_iterations - 1:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_iter_{i}_dim_{embedding_dim}.pt')
        save_checkpoint(paper_dict, predictions, i, ckpt_path, l_next, new_optimizer)

        saved_checkpoints.append(ckpt_path)

        # Remove older checkpoints if more than max_saved
        if len(saved_checkpoints) > max_saved:
            old_files = saved_checkpoints.pop(0)  # Get the oldest checkpoint
            if os.path.exists(old_files):
                os.remove(old_files) 

        items = sorted(predictions.items())
        y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
        y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

        evaluator = Evaluator(name='ogbn-mag')
        result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        acc.append(result['acc'])
        wandb.log({"Accuracy": result['acc']})


save_dir = f'dataset/ogbn_mag/processed/Predictions'
os.makedirs(save_dir, exist_ok=True)
torch.save(predictions, os.path.join(save_dir, f'pred_dict_{embedding_dim}_manhattan.pt'))

# Filter paper_dict to only include papers for which predictions were made
filtered_paper_dict = {
    'paper': {k: v for k, v in paper_dict['paper'].items() if k in predictions},
    'venue': paper_dict['venue']
}
num_papers = len(filtered_paper_dict['paper'])

torch.save(filtered_paper_dict, os.path.join(save_dir, f'valid_embeddings_dict_{embedding_dim}_manhattan.pt'))

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=filtered_paper_dict,sample_size=1000,filename='dataset/ogbn_mag/processed/Predictions/plot_valid_top_5',dim=embedding_dim,top=5,plot_params={
        "alpha": alpha,
        "lambda": lam,
        "lr": lr,
        "epochs": num_epochs,
        "dim": embedding_dim
    })

plot_paper_venue_embeddings(venue_value=venue_value,embed_dict=filtered_paper_dict,sample_size=1000,filename='dataset/ogbn_mag/processed/Predictions/plot_valid_top_1',dim=embedding_dim,top=1,plot_params={
        "alpha": alpha,
        "lambda": lam,
        "lr": lr,
        "epochs": num_epochs,
        "dim": embedding_dim
    })



# print(counter)

# plt.plot(range(1, len(acc) + 1), acc, marker='o')
# plt.xlabel('Run Number')
# plt.ylabel('Accuracy')
# plt.title('Accuracy over Runs')
# plt.grid(True)
# plt.show()