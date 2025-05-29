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