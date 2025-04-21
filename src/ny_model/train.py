# train.py

from ny_model.load_128 import load_data, MLPClassifier, train_and_evaluate
import torch

if __name__ == "__main__":
    # Load datasets
    train_dataset, valid_dataset, test_dataset, y = load_data()

    # Model hyperparameters
    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(torch.unique(y))

    # Initialize model
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, y=y)

    # Train and evaluate
    train_and_evaluate(model, train_dataset, valid_dataset, test_dataset)
