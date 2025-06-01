# train.py
import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import datetime

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

# dir_path = '/work3/s224225/'

# if os.path.isdir(dir_path):
#     os.chdir(dir_path)  # change working directory
#     # Set root directory for all PyTorch Lightning outputs
#     PL_ROOT = os.getenv('PL_ROOT', os.path.join(dir_path, 'pytorch_lightning'))
#     # Ensure the directory exists
#     os.makedirs(PL_ROOT, exist_ok=True)
# else:
#     print(f"Directory {dir_path} does not exist. Skipping setup.")


def main():
    lr = 0.00001
    batch_size = 128
    num_epochs = 75
    dropout_rate = 0.2
    hidden_dim = 4096


    wandb.init(    project="Bachelor_projekt",
    name=f"MLP model run {datetime.datetime.now()}",
    config={
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "dropout_rate": dropout_rate,
        "hidden_dim": hidden_dim}
        )

    X_train, y_train, X_valid, y_valid, X_test, y_test, y, _, _, _ = prep_data()

    data_module = VenueDataModule(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=batch_size
    )

    # model = VenueClassifier(
    #     y_train=y_train,
    #     input_dim=128,
    #     hidden_dim=config.hidden_dim,
    #     num_classes=y.max().item() + 1,
    #     lr=config.lr,
    #     dropout_rate=config.dropout_rate
    # )


    model = VenueClassifier(
        y_train=y_train,
        input_dim=128,
        hidden_dim=hidden_dim,
        num_classes=y.max().item() + 1,
        lr=lr,
        dropout_rate=dropout_rate
    )
    
    wandb_logger = WandbLogger(project="Bachelor_projekt", log_model=True)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',    # metric to monitor
        patience=5,            # number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'             # 'min' since lower val_loss is better
    )

    OUTPUT_DIR = "src/MLP/lightning_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        default_root_dir=OUTPUT_DIR
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    model.plot_metrics()

    final_ckpt_path = os.path.join(OUTPUT_DIR, "final_model.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")

    test_loader = data_module.test_dataloader()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch)

    # concatenate everything into single tensor, then squeeze to 1-D
    all_preds = torch.cat(all_preds, dim=0).view(-1).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).view(-1).cpu().numpy()

    # C) write preds & true labels to CSV
    csv_out = os.path.join(OUTPUT_DIR, "test_predictions_mixed_data.csv")
    df = pd.DataFrame({
        "true_label": all_labels,
        "pred_label": all_preds
    })
    df.to_csv(csv_out, index=False)

    print(f"Saved test predictions & true labels to {csv_out}")  


if __name__ == "__main__":
    main()

