import torch
import pytorch_lightning as pl
from vector_128_model import VenueDataModule, VenueClassifier, prep_data
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision('medium')

# Training with PyTorch Lightning

X_train, y_train, X_valid, y_valid, X_test, y_test, y = prep_data()

data_module = VenueDataModule(X_train, y_train, X_valid, y_valid, X_test, y_test)
model = VenueClassifier(y, input_dim=128, num_classes=349, lr=0.01, dropout_rate = 0.1)
if torch.cuda.is_available():
    print("Using GPU for training")
    
trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=1)  # Set accelerator to "gpu" if using CUDA
trainer.fit(model, datamodule=data_module)

# Testing the Model
trainer.test(datamodule=data_module)


model.plot_metrics()