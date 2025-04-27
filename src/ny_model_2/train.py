import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# Load your preprocessed data
from ny_model_2.load import load_data  # replace with the actual filename if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_dataset, valid_dataset, test_dataset, y_all = load_data()

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2048)
test_loader = DataLoader(test_dataset, batch_size=2048)


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=349):  # 349 classes for ogbn-mag
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


if __name__ == "__main__":
    # Load data
    train_dataset, valid_dataset, test_dataset, y_all = load_data()

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=2048)
    test_loader = DataLoader(test_dataset, batch_size=2048)

    # Model
    model = MLPClassifier(output_dim=y_all.max().item() + 1)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",  # uses GPU if available
        devices=1 if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
        log_every_n_steps=10
    )

    # Train and test
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)