# train.py
import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data
from pytorch_lightning.loggers import WandbLogger
import wandb

torch.set_float32_matmul_precision('medium')

def main():
    wandb.init()
    config = wandb.config

    X_train, y_train, X_valid, y_valid, X_test, y_test, y = prep_data()

    data_module = VenueDataModule(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=config.batch_size
    )

    model = VenueClassifier(
        y_train=y_train,
        input_dim=128,
        hidden_dim=config.hidden_dim,
        num_classes=y.max().item() + 1,
        lr=config.lr,
        dropout_rate=config.dropout_rate
    )

    wandb_logger = WandbLogger(project="Bachelor_projekt", log_model=True)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    model.plot_metrics()

if __name__ == "__main__" and not wandb.run:
    main()
