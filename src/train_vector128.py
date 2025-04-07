import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data

torch.set_float32_matmul_precision('medium')

# --------------------------------
# Training with PyTorch Lightning
# --------------------------------

X_train, y_train, X_valid, y_valid, X_test, y_test = prep_data()

data_module = VenueDataModule(X_train, y_train, X_valid, y_valid, X_test, y_test)
model = VenueClassifier(input_dim=128, num_classes=349, lr=0.001)
if torch.cuda.is_available():
    print("Using GPU for training")
trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=1)  # Set accelerator to "gpu" if using CUDA
trainer.fit(model, datamodule=data_module)

# --------------------------------
# Testing the Model
# --------------------------------
trainer.test(datamodule=data_module)