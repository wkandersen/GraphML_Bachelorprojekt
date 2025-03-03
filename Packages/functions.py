import torch

def loss_function(z, datamatrix_tensor, alpha=1.0, eps=1e-8):
    sum_loss = 0
    for entry in datamatrix_tensor:
        label, u_idx, v_idx = entry
        z_u = z[u_idx]
        z_v = z[v_idx]
        prob = edge_probability(z_u, z_v, alpha)

        # Numerical stability: clamp probabilities to avoid log(0)
        prob = torch.clamp(prob, eps, 1 - eps)

        sum_loss += label.float() * torch.log(prob) + (1 - label.float()) * torch.log(1 - prob)

    return -sum_loss / len(datamatrix_tensor)

def loss_function(z, datamatrix_tensor, alpha=1.0, eps=1e-8):
    sum_loss = 0
    for entry in datamatrix_tensor:
        label, u_idx, v_idx = entry
        z_u = z[u_idx]
        z_v = z[v_idx]
        prob = edge_probability(z_u, z_v, alpha)

        # Numerical stability: clamp probabilities to avoid log(0)
        prob = torch.clamp(prob, eps, 1 - eps)

        sum_loss += label.float() * torch.log(prob) + (1 - label.float()) * torch.log(1 - prob)

    return -sum_loss / len(datamatrix_tensor)

# 5️ Define Loss Function (Using Data Matrix)
def edge_probability(z_i, z_j, alpha=1.0):
    dist = torch.norm(z_i - z_j) ** 2  # Squared Euclidean distance using PyTorch
    return 1 / (1 + torch.exp(-alpha + dist))  # Logistic function using PyTorch

