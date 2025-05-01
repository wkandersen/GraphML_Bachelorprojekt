import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Prediction:
    def __init__(self,alpha,paper_dict,valid_dict,venue_dict,venue_value):
        self.alpha = alpha
        self.paper_dict = paper_dict
        self.valid_dict = valid_dict
        self.venue_dict = venue_dict
        self.venue_value = venue_value

    def predict(self,input_dict):
        predictions = {}
        for x, y in input_dict.items():
                alpha = self.alpha
                logi_f = []

                for i in range(len(self.venue_dict)):
                        dist = torch.norm(y - self.venue_dict[i])**2  # Euclidean distance
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
                predictions[x] = (int(self.venue_value[x].numpy()),predicted_node_id)
        return predictions
    
    def accuracy(self):
        predictions_train = self.predict(self.paper_dict)
        predictions_valid = self.predict(self.valid_dict)
        # predictions_test = self.predict(self.valid_dict)
        acc_train = self.evaluate_predictions(predictions_train)
        acc_valid = self.evaluate_predictions(predictions_valid)
        return acc_train,acc_valid
    
    def evaluate_predictions(self,predictions):
        evaluator = Evaluator(name='ogbn-mag')
        items = sorted(predictions.items())

        y_true = torch.tensor([true_pred[0] for _, true_pred in items]).view(-1, 1)
        y_pred = torch.tensor([true_pred[1] for _, true_pred in items]).view(-1, 1)
        
        result = evaluator.eval({
            'y_true': y_true,
            'y_pred': y_pred,
        })
        return result
