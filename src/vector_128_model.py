import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
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

    return X_train, y_train, X_valid, y_valid, X_test, y_test





class VenueDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=64):
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

# --------------------------------
# PyTorch Lightning Model
# --------------------------------
class VenueClassifier(pl.LightningModule):
    def __init__(self, input_dim=128, num_classes=349, lr=0.001):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.squeeze())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.squeeze())
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.squeeze())
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
