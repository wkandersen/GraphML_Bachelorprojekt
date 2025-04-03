import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

valid_dict = torch.load("dataset/ogbn_mag/processed/hpc/valid_dict.pt")
venue_dict = torch.load("dataset/ogbn_mag/processed/hpc/venue_dict.pt")
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt")
predictions = {}

for x, y in valid_dict.items():
        alpha = 0.001
        logi_f = []

        for i in range(len(venue_dict)):
                dist = torch.norm(y - venue_dict[i])**2  # Euclidean distance
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
        predictions[x] = (int(venue_value[x].numpy()),predicted_node_id)

# Accuarcy calculation
correct_predictions = 0
for i in predictions:
    if predictions[i][0] == predictions[i][1]:
        correct_predictions += 1
accuracy = correct_predictions / len(predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

true_labels = [value[0] for value in predictions.values()]
predicted_labels = [value[1] for value in predictions.values()]

# Get all unique labels (ensures matrix includes all possible classes)
all_labels = sorted(set(true_labels) | set(predicted_labels))  # Union of true & predicted labels

# Compute confusion matrix with explicit labels
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

# Print classification report
print(classification_report(true_labels, predicted_labels, labels=all_labels))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("dataset/ogbn_mag/processed/hpc/confusion_matrix.png") 
plt.close()
