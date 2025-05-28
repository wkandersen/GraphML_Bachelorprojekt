import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

emb_dim = 2
predictions = torch.load(f'dataset/ogbn_mag/processed/Predictions/pred_dict_2.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(predictions)
# Accuarcy calculation
correct_predictions = 0
for pred in predictions.values():
    if pred[0] == pred[1]:
        correct_predictions += 1
accuracy = correct_predictions / len(predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

items = sorted(predictions.items())
true_labels = np.array([value[0] for value in predictions.values()])
predicted_labels = np.array([value[1] for value in predictions.values()])

y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)

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
# plt.show()
plt.savefig("dataset/ogbn_mag/processed/Predictions/confusion_matrix.png") 
plt.close()

from ogb.nodeproppred import Evaluator
evaluator = Evaluator(name='ogbn-mag')
result = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })

print(f'Accuracy using evaluator: {result}')