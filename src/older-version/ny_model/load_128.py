# model_setup.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import pytorch_lightning as pl
import numpy as np


def load_data():
    try:
        data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip', header=None)
        data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip', header=None)
        data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip', header=None)
    except FileNotFoundError:
        os.chdir("..")
        data_train = pd.read_csv('dataset/ogbn_mag/split/time/paper/train.csv.gz', compression='gzip', header=None)
        data_valid = pd.read_csv('dataset/ogbn_mag/split/time/paper/valid.csv.gz', compression='gzip', header=None)
        data_test = pd.read_csv('dataset/ogbn_mag/split/time/paper/test.csv.gz', compression='gzip', header=None)

    data, _ = torch.load(r"dataset/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)

    X = data.x_dict[('paper')]
    y = data['y_dict']['paper']

    nums_valid = torch.tensor(data_valid[0])
    nums_test = torch.tensor(data_test[0])
    nums_train = torch.tensor(data_train[0])

    X_train, y_train = X[nums_train], y[nums_train]
    X_valid, y_valid = X[nums_valid], y[nums_valid]
    X_test, y_test = X[nums_test], y[nums_test]

    train_dataset = TensorDataset(X_train, y_train.squeeze())
    valid_dataset = TensorDataset(X_valid, y_valid.squeeze())
    test_dataset = TensorDataset(X_test, y_test.squeeze())


    return train_dataset, valid_dataset, test_dataset, y


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3, lr=0.001, y=None):
        super().__init__()
        self.lr = lr

        if y is not None:
            labels_y = y.flatten().cpu().numpy()
            class_weights = compute_class_weight('balanced', classes=np.unique(labels_y), y=labels_y)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)  # final output layer (no activation here)
        )


        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if batch_idx == 0:  # only log once per epoch to avoid clutter
            topk_vals, topk_idxs = torch.topk(outputs, k=5, dim=1)  # top 5 classes
            print(f"Top-5 probs: {topk_vals[0].softmax(dim=0)}")  # convert logits to probs
            print(f"Top-5 classes: {topk_idxs[0]}")
            print(f"True label: {labels[0]}")
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def train_and_evaluate(model, train_dataset, valid_dataset, test_dataset, batch_size=512, max_epochs=50):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu", devices=1)

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    return model
