import torch
import sys
import os
import gc
from Packages.mini_batches import mini_batches_code
from Packages.data_divide import paper_c_paper_train, paper_c_paper_valid, data
import wandb
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from ogb.nodeproppred import Evaluator
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

class LossFunction:
    def __init__(self, alpha=1.0, eps=1e-8, use_regularization=False, lam=0.01):
        self.alpha = alpha
        self.eps = eps
        self.use_regularization = use_regularization
        self.lam = lam

    def edge_probability(self, z_i, z_j):
        dist_sq = torch.sum((z_i - z_j) ** 2, dim=1)
        return torch.sigmoid(self.alpha - dist_sq)

    def link_loss(self, label, z_u, z_v):
        prob = self.edge_probability(z_u, z_v)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)
        return label * torch.log(prob) + (1 - label) * torch.log(1 - prob)

    def compute_loss(self, z, datamatrix_tensor):
        labels = datamatrix_tensor[:, 0].float()
        u_idx = datamatrix_tensor[:, 1].long()
        v_idx = datamatrix_tensor[:, 2].long()
        pv1_idx = datamatrix_tensor[:, 3].long()
        pv2_idx = datamatrix_tensor[:, 4].long()

        edge_entities = {
            0: 'paper',
            1: 'author',
            2: 'institution',
            3: 'field_of_study',
            4: 'venue'
        }

        z_u_list = []
        z_v_list = []
        new_labels = []
        missing = []

        for label, i_u, i_v, t_u, t_v in zip(labels, u_idx, v_idx, pv1_idx, pv2_idx):
            ent_u = edge_entities[t_u.item()]
            ent_v = edge_entities[t_v.item()]
            idx_u = i_u.item()
            idx_v = i_v.item()

            if idx_u in z[ent_u] and idx_v in z[ent_v]:
                z_u_list.append(z[ent_u][idx_u])
                z_v_list.append(z[ent_v][idx_v])
                new_labels.append(label)
            else:
                missing.extend([idx_u, idx_v])

        if not z_u_list or not z_v_list:
            raise ValueError("No valid pairs found for z_u and z_v.")

        z_u = torch.stack(z_u_list).to(device)
        z_v = torch.stack(z_v_list).to(device)
        labels = torch.tensor(new_labels, dtype=torch.float).to(device)

        link_loss = self.link_loss(labels, z_u, z_v)
        loss = -torch.mean(link_loss)

        if self.use_regularization:
            regularization = self.lam * torch.sum(z ** 2)
            loss += regularization

        return loss, set(missing)

venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device)
log_path = "dataset/ogbn_mag/processed/Predictions/exceptions_log.txt"

lr = 0.1
num_epochs = 100
alpha = 0.1
eps = 0.001
lam = 0.01
batch_size = 1
num_iterations = len(paper_c_paper_valid[0])
emb_dim = 2

run = wandb.init(
    project="Bachelor_projekt",
    name=f"embed_valid_run_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "alpha": alpha,
        "lam": lam,
        "emb_dim": emb_dim
    },
)

check = torch.load(f'checkpoint/checkpoint_iter_75_{emb_dim}_3_epoch_3.pt', map_location=device)
paper_dict = check['collected_embeddings']

# Move all embeddings to device
for ent_type in paper_dict:
    for k, v in paper_dict[ent_type].items():
        paper_dict[ent_type][k] = v.to(device)

loss_function = LossFunction(alpha=alpha, eps=eps, use_regularization=False)

unique_train = set(paper_c_paper_train.flatten().unique().tolist())
unique_valid = set(paper_c_paper_valid.flatten().unique().tolist())
valid_exclusive = unique_valid - unique_train
l_prev = list(valid_exclusive)
predictions = {}

for i in range(num_iterations):
    try:
        mini_b = mini_batches_code(paper_c_paper_valid, l_prev, batch_size, ('paper', 'cites', 'paper'), data)
        dm, l_next, random_sample = mini_b.data_matrix()
        if len(dm) < 2:
            print(f"[SKIP] Too few nodes for venue comparison: {random_sample}")
            continue

        dm = dm[dm[:, 4] != 4]
        print(dm)

        mini_btrain1 = mini_batches_code(paper_c_paper_train, dm[:, 2].tolist(), len(dm[:, 2]), ('paper', 'cites', 'paper'), data)
        dmtrain1, ultrain1, random_sampletrain1 = mini_btrain1.data_matrix()

        test1 = dmtrain1[dmtrain1[:, 4] != 4]
        test2 = test1[test1[:, 0] == 1]
        list_test = test2[:, 2].unique().tolist()

        concat_dm = torch.cat((dm, dmtrain1), 0)
        rows = [[0, random_sample[0], test_item, 0, 0] for test_item in list_test]
        tensor_result = torch.tensor(rows).to(device)
        concat_dm = concat_dm.to(device)
        tensor_result = tensor_result.to(device)
        final = torch.cat((concat_dm, tensor_result), 0).to(device)

        embedding_list = []
        for test_item in list_test:
            if test_item in paper_dict['paper']:
                embedding_list.append(paper_dict['paper'][test_item])

        if len(embedding_list) == 0:
            new_embedding = torch.nn.Parameter(torch.randn(2, device=device, requires_grad=True))
        else:
            mean_tensor = torch.mean(torch.stack(embedding_list), dim=0)
            new_embedding = torch.nn.Parameter(mean_tensor.clone().detach().to(device).requires_grad_())
            print(f"Element-wise mean tensor: {mean_tensor}")

        paper_dict['paper'][random_sample[0]] = new_embedding
        print(paper_dict['paper'][random_sample[0]])
        new_optimizer = torch.optim.Adam([paper_dict['paper'][random_sample[0]]], lr=lr)

        prev_losses = []
        patience = 5
        tolerance = 1e-4  # Define how close "very close" means

        for epoch in range(num_epochs):
            new_optimizer.zero_grad()
            loss, missing = loss_function.compute_loss(paper_dict, final)
            loss.backward()
            new_optimizer.step()
            
            current_loss = loss.detach().item()
            wandb.log({"loss": current_loss})
            
            # Track loss history
            prev_losses.append(current_loss)

            if len(prev_losses) > patience:
                recent = prev_losses[-patience:]
                if max(recent) - min(recent) < tolerance:
                    print(f"[EARLY STOP] Loss converged after {epoch + 1} epochs.")
                    break



        wandb.log({"final_loss": loss.detach().item()})
        wandb.log({"Sample number": int(i)})
        if len(missing) != 0:
            print(f'Missing: {missing}')

        print(paper_dict['paper'][random_sample[0]])
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

        wandb.log({"true_class_rank": true_class_rank})


        items = sorted(predictions.items())
        y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
        y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

        evaluator = Evaluator(name='ogbn-mag')
        result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        wandb.log({"Accuracy": result})

        l_prev = l_next
        new_embedding = new_embedding.detach()
        del new_optimizer

        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        error_message = f"[ERROR] Iteration {i} failed due to: {e}\n"
        stack_trace = traceback.format_exc()

        print(error_message)
        print(stack_trace)

        with open(log_path, "a") as log_file:
            log_file.write(error_message)
            log_file.write(stack_trace)
            log_file.write("\n" + "-" * 80 + "\n")

        continue

save_dir = f'dataset/ogbn_mag/processed/Predictions'
os.makedirs(save_dir, exist_ok=True)
torch.save(predictions, os.path.join(save_dir, f'pred_dict_{emb_dim}.pt'))

print('finish')
