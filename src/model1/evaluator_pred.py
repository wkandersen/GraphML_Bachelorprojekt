import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

paper_dict = torch.load("dataset/ogbn_mag/processed/hpc/paper_dict.pt", map_location=device)
valid_dict = torch.load("dataset/ogbn_mag/processed/hpc/valid_dict.pt", map_location=device)
venue_dict = torch.load("dataset/ogbn_mag/processed/hpc/venue_dict.pt", map_location=device)
venue_value = torch.load("dataset/ogbn_mag/processed/venue_value.pt", map_location=device)

# Prediction function
def predict(input_dict):
    predictions = {}
    for x, y in input_dict.items():
            alpha = 0.001
            logi_f = []

            for i in range(len(venue_dict)):
                    dist = torch.norm(y - venue_dict[i])**2  # Euclidean distance
                    logi = 1 / (1 + torch.exp(alpha + dist))  # Logistic function
                    logi_f.append((logi.item(), i))  # Store tuple (probability, node ID)

            # Separate values for softmax computation
            logits, node_ids = zip(*logi_f)  # Unzips into two lists

            # Convert logits to a tensor and apply softmax
            logi_f_tensor = torch.tensor(logits).to(device)
            softma = F.softmax(logi_f_tensor, dim=0)

            # Get the index of the highest probability
            high_prob_idx = torch.argmax(softma).item()

            # Get the corresponding node ID and its softmax probability
            predicted_node_id = node_ids[high_prob_idx]
            predictions[x] = (int(venue_value[x].numpy()),predicted_node_id)
    return predictions

# Run predictions
predictions_valid = predict(valid_dict)
predictions_test = predict(paper_dict)

@torch.no_grad()
def evaluate_predictions(predictions):
    evaluator = Evaluator(name='ogbn-mag')
    items = sorted(predictions.items())

    y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
    y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)
    
    result = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })
    return result

acc_test = evaluate_predictions(predictions_test)
acc_valid = evaluate_predictions(predictions_valid)