import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embed_batches import venue_dict
from src.embed_valid_sample import new_embedding

alpha = 0.001
logi_f = []

for i in range(len(venue_dict)):
        dist = torch.norm(new_embedding.weight - venue_dict[i])**2  # Euclidean distance
        logi = 1 / (1 + torch.exp(alpha + dist))  # Logistic function
        logi_f.append((logi.item(), i))  # Store tuple (probability, node ID)

# Separate values for softmax computation
logits, node_ids = zip(*logi_f)  # Unzips into two lists

# Convert logits to a tensor and apply softmax
logi_f_tensor = torch.tensor(logits)
softma = F.softmax(logi_f_tensor, dim=0)

# Get the index of the highest probability
high_prob_idx = torch.argmax(softma).item()

# Get the corresponding node ID and its softmax probability
predicted_node_id = node_ids[high_prob_idx]
highest_prob_value = softma[high_prob_idx].item()

# Print the results
print(f"Predicted Node ID: {predicted_node_id}")
print(f"Highest Softmax Probability: {highest_prob_value}")
