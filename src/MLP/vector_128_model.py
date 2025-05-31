
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



    return X_train, y_train, X_valid, y_valid, X_test, y_test, y




class VenueDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = TensorDataset(X_train, y_train)
        self.valid_dataset = TensorDataset(X_valid, y_valid)
        self.test_dataset = TensorDataset(X_test, y_test)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

#MODEL
class VenueClassifier(pl.LightningModule):
    def __init__(self, y_train, input_dim=128, hidden_dim=256, num_classes=349, lr=0.001, dropout_rate=0.1):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters(ignore=['y_train'])
        self.train_losses = []
        self.test_losses = []
        self.epoch_train_losses = []
        self.epoch_test_losses = []

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        self.criterion = nn.NLLLoss()


        hidden_dim2 = hidden_dim // 2
        hidden_dim3 = hidden_dim2 // 2

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim3, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.epoch_train_losses.append(loss.item())

        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean(self.epoch_train_losses)
        self.train_losses.append(avg_loss)
        self.epoch_train_losses.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("test_acc", acc)

        self.epoch_test_losses.append(loss.item())

    def on_test_epoch_end(self):
        avg_loss = np.mean(self.epoch_test_losses)
        self.test_losses.append(avg_loss)
        self.epoch_test_losses.clear()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def plot_metrics(self, path="Plots/128_vector"):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.train_losses)), self.train_losses, label="Training Loss")
        plt.plot(range(len(self.test_losses)), self.test_losses, label="Test Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Test Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/losses_train_vs_test.png")
        plt.close()
