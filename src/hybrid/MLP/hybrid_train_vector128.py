# train.py
import torch
import pytorch_lightning as pl
from hybrid_vector_128_model import VenueDataModule, VenueClassifier, prep_data
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
torch.set_float32_matmul_precision('medium')

# dir_path = "/path/to/some/directory"
# # Set root directory for all PyTorch Lightning outputs
# PL_ROOT = os.getenv('PL_ROOT', '/work3/s224225/pytorch_lightning')
# # Ensure the directory exists
# os.makedirs(PL_ROOT, exist_ok=True)


def main():
    # wandb.init()
    # config = wandb.config

    X_train, y_train, X_valid, y_valid, X_test, y_test, y, ids_train, ids_valid, ids_test = prep_data()


    data_module = VenueDataModule(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=256
    )


    model = VenueClassifier(
        y_train=y_train,
        input_dim=128,
        hidden_dim=512,
        num_classes=y.max().item() + 1,
        lr=0.001,
        dropout_rate=0.2
    )
    # wandb_logger = WandbLogger(project="Bachelor_projekt", log_model=True)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    # Extract 2D embeddings after training
    model.eval()
    with torch.no_grad():
        embeddings_train = model.get_embedding(X_train).cpu()
        embeddings_valid = model.get_embedding(X_valid).cpu()
        embeddings_test = model.get_embedding(X_test).cpu()

    # Save or visualize embeddings
    import os
    import numpy as np

    os.makedirs("src/hybrid/MLP/embeddings", exist_ok=True)
    # Convert paper IDs to ints for dictionary keys
    embedding_dict_train = {int(pid):  emb.detach().clone() for pid, emb in zip(ids_train, embeddings_train)}
    embedding_dict_valid = {int(pid): emb.detach().clone() for pid, emb in zip(ids_valid, embeddings_valid)}
    embedding_dict_test = {int(pid): emb.detach().clone() for pid, emb in zip(ids_test, embeddings_test)}

    torch.save(embedding_dict_train, f"src/hybrid/MLP/embeddings/train_embeddings_dict_max_epochs{trainer.max_epochs}.pt")
    torch.save(embedding_dict_valid, f"src/hybrid/MLP/embeddings/valid_embeddings_dict_max_epochs{trainer.max_epochs}.pt")
    torch.save(embedding_dict_test, f"src/hybrid/MLP/embeddings/test_embeddings_dict_max_epochs{trainer.max_epochs}.pt")

    print("2D embeddings saved to the 'embeddings' folder.")

    model.plot_metrics()

if __name__ == "__main__":
    main()

