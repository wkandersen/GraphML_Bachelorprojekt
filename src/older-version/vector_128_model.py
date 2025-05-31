import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set working directory

def prep_data():
    try:
        data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
        data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
        data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)
    except FileNotFoundError:
        os.chdir("..")
        data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip',header = None)
        data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip',header = None)
        data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip',header = None)

    data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

    # Extract edges for "paper" -> "cites" -> "paper"
    X = data.x_dict[('paper')]
    y = data['y_dict']['paper']


    # Unique paper IDs to keep (Ensure it's a PyTorch tensor)
    nums_valid = torch.tensor(data_valid[0])
    nums_test = torch.tensor(data_test[0])
    nums_train = torch.tensor(data_train[0])


    # Filter the dataset using indices
    X_train, y_train = X[nums_train], y[nums_train]
    X_valid, y_valid = X[nums_valid], y[nums_valid]
    X_test, y_test = X[nums_test], y[nums_test]

    y_train = y[nums_train].long()
    y_valid = y[nums_valid].long()
    y_test = y[nums_test].long()

    train_mean = X_train.mean(dim=0)
    train_std = X_train.std(dim=0)
    train_std[train_std < 1e-6] = 1.0  # Prevent divide by 0

    X_train = (X_train - train_mean) / train_std
    X_valid = (X_valid - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std


    return X_train, y_train, X_valid, y_valid, X_test, y_test, y





class VenueDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=2048):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = TensorDataset(X_train, y_train)
        self.valid_dataset = TensorDataset(X_valid, y_valid)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

#MODEL
class VenueClassifier(pl.LightningModule):
    def __init__(self, y, input_dim=128, num_classes=349, lr=0.01, dropout_rate = 0.3):
        super().__init__()
        self.lr = lr
        labels_y = y.flatten().cpu().numpy()
        # Initialize metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        # Compute class weights
        # self._raw_class_weights = compute_class_weight('balanced', classes=np.unique(labels_y), y=labels_y)
        # self.register_buffer("class_weights", torch.tensor(self._raw_class_weights, dtype=torch.float32))  # buffer = auto-device-safe
       
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()  # Will be on correct device later

    def forward(self, x):
        return self.model(x)
    
    # def on_fit_start(self):
        # This guarantees it's on the same device as the model
        # self.criterion.weight = self.class_weights.to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss.item())
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)  # ensure y is 1D
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("val_acc", acc, prog_bar=True)
        self.val_losses.append(loss.item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)  # ensure y is 1D
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        # Plotting method after training
    def plot_metrics(self, path="Plots/128_vector"):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        # Plot Losses
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.savefig(f"{path}/losses_plot.png")
        plt.tight_layout()

        plt.close()  # Close the figure to release memory
