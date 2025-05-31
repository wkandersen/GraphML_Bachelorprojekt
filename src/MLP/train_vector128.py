# train.py
import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
torch.set_float32_matmul_precision('medium')

dir_path = '/work3/s224225/'

if os.path.isdir(dir_path):
    os.chdir(dir_path)  # change working directory
    # Set root directory for all PyTorch Lightning outputs
    PL_ROOT = os.getenv('PL_ROOT', os.path.join(dir_path, 'pytorch_lightning'))
    # Ensure the directory exists
    os.makedirs(PL_ROOT, exist_ok=True)
else:
    print(f"Directory {dir_path} does not exist. Skipping setup.")


def main():
    wandb.init()
    config = wandb.config

    X_train, y_train, X_valid, y_valid, X_test, y_test, y = prep_data()

    data_module = VenueDataModule(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=256
    )

    model = VenueClassifier(
        y_train=y_train,
        input_dim=128,
        hidden_dim=config.hidden_dim,
        num_classes=y.max().item() + 1,
        lr=config.lr,
        dropout_rate=config.dropout_rate
    )


    # model = VenueClassifier(
        # y_train=y_train,
        # input_dim=128,
        # hidden_dim=512,
        # num_classes=y.max().item() + 1,
        # lr=0.001,
        # dropout_rate=0.2
    # )
    wandb_logger = WandbLogger(project="Bachelor_projekt", log_model=True)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    model.plot_metrics()

if __name__ == "__main__":
    main()

