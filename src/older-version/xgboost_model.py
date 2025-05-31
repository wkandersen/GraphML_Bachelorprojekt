import xgboost as xgb
import numpy as np
import logging
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from vector_128_model import prep_data

class XGBoostModel:
    def __init__(self, num_classes=349, max_depth=8, learning_rate=0.1, num_boost_round=300, early_stopping_rounds=10, device='cuda'):
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Load the data
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, _ = prep_data()

        # Convert data to DMatrix
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dvalid = xgb.DMatrix(self.X_valid, label=self.y_valid)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # Set up XGBoost parameters
        self.params = {
            'objective': 'multi:softmax',
            'num_class': self.num_classes,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'eval_metric': ['mlogloss', 'merror'],
            'tree_method': 'hist',
            'lambda': 1.0,   # L2 regularization term
            'alpha': 0.1,    # L1 regularization term
            'device': self.device,
        }

        # List of evaluation sets
        self.evals = [(self.dtrain, 'train'), (self.dvalid, 'eval')]
        self.evals_result = {}

    def train(self):
        self.logger.info("Training started...")
        
        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=self.dtrain,
            num_boost_round=self.num_boost_round,
            evals=self.evals,
            evals_result=self.evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=True
        )
        
        self.logger.info("Training completed.")
        self.logger.info(f"Best training log-loss: {self.evals_result['train']['mlogloss'][-1]}")
        self.logger.info(f"Best validation log-loss: {self.evals_result['eval']['mlogloss'][-1]}")

    def predict(self):
        self.logger.info("Prediction started...")
        
        y_pred = self.model.predict(self.dtest)
        return y_pred

    def evaluate(self, y_pred):
        self.logger.info("Evaluating model...")
        
        accuracy = accuracy_score(self.y_test, y_pred)
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_metrics(self):
        self.logger.info("Plotting metrics...")

        # Plot Log Loss
        plt.figure(figsize=(12, 6))
        plt.plot(self.evals_result['train']['mlogloss'], label='Train Log Loss')
        plt.plot(self.evals_result['eval']['mlogloss'], label='Validation Log Loss')
        plt.title('Log Loss per Round')
        plt.xlabel('Boosting Round')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.savefig(f"Plots/xgboost_losses_plot_2.png")

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(1 - np.array(self.evals_result['train']['merror']), label='Train Accuracy')
        plt.plot(1 - np.array(self.evals_result['eval']['merror']), label='Validation Accuracy')
        plt.title('Accuracy per Round')
        plt.xlabel('Boosting Round')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"Plots/xgboost_accuracy_plot_2.png")

# Example of usage:
if __name__ == "__main__":
    model = XGBoostModel(num_classes=349, max_depth=10, learning_rate=0.01)
    model.train()  # Train the model
    y_pred = model.predict()  # Make predictions
    accuracy = model.evaluate(y_pred)  # Evaluate accuracy
    model.plot_metrics()  # Plot the metrics
