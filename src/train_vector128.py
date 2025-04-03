import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data


# --------------------------------
# Training with PyTorch Lightning
# --------------------------------

X_train, y_train, X_valid, y_valid, X_test, y_test, y = prep_data()

data_module = VenueDataModule(X_train, y_train, X_valid, y_valid, X_test, y_test)
model = VenueClassifier(num_classes=torch.unique(y))

trainer = pl.Trainer(max_epochs=20, accelerator="auto", devices=1)  # Set accelerator to "gpu" if using CUDA
trainer.fit(model, datamodule=data_module)

# --------------------------------
# Testing the Model
# --------------------------------
trainer.test(datamodule=data_module)