import torch
import numpy as np
import matplotlib as plt
import seaborn as sns
from functions import edge_probability

class Evaluator:
    def __init__(self, all_nodes, num_nodes_set_1, emb_matrix, threshold, edges):
        self.all_nodes = all_nodes
        self.num_nodes_set_1 = num_nodes_set_1
        self.emb_matrix = emb_matrix
        self.threshold = threshold
        self.edges = edges

    def compute_TP_FP_TN_FN(self):
        TP = FP = TN = FN = 0
        
        with torch.no_grad():
            for u in range(len(self.all_nodes)):
                for v in range(u + 1, len(self.all_nodes)):
                    if (u in range(self.num_nodes_set_1) and v in range(self.num_nodes_set_1, len(self.all_nodes))) or \
                       (v in range(self.num_nodes_set_1) and u in range(self.num_nodes_set_1, len(self.all_nodes))):
                        
                        label = 1 if (self.all_nodes[u], self.all_nodes[v]) in self.edges or \
                                       (self.all_nodes[v], self.all_nodes[u]) in self.edges else 0
                        
                        u_embedding = torch.tensor(self.emb_matrix[u], dtype=torch.float32)
                        v_embedding = torch.tensor(self.emb_matrix[v], dtype=torch.float32)
                        prob = edge_probability(u_embedding, v_embedding, alpha=1.0)
                        
                        predicted_label = 1 if prob > self.threshold else 0
                        
                        if label == 1 and predicted_label == 1:
                            TP += 1
                        elif label == 0 and predicted_label == 1:
                            FP += 1
                        elif label == 0 and predicted_label == 0:
                            TN += 1
                        elif label == 1 and predicted_label == 0:
                            FN += 1
        
        return TP, FP, TN, FN

    def compute_accuracy(self):
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN()
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    def compute_recall(self):
        TP, _, _, FN = self.compute_TP_FP_TN_FN()
        return TP / (TP + FN) if (TP + FN) > 0 else 0
    
    def compute_precision(self):
        TP, FP, _, _ = self.compute_TP_FP_TN_FN()
        return TP / (TP + FP) if (TP + FP) > 0 else 0
    
    def f1_score(self):
        recall, precision = self.compute_recall(), self.compute_precision()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    def plot_confusion_matrix(self):
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN()
        conf_matrix = np.array([[TN, FP], 
                        [FN, TP]])

        # Plot the confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"])

        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.show()
    

