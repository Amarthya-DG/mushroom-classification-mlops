import json
import logging
import os
import sys
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.data_processing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, experiment_name="mushroom-classification"):
        """
        Initialize the ModelTrainer.

        Args:
            experiment_name (str, optional): Name of the MLflow experiment. Defaults to "mushroom-classification".
        """
        self.experiment_name = experiment_name
        self.data_processor = DataProcessor()
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
        }

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        # Create directories for model artifacts
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/plots", exist_ok=True)

    def train_and_evaluate(self):
        """
        Train and evaluate multiple models, tracking results with MLflow.

        Returns:
            str: Name of the best performing model.
        """
        logger.info("Starting model training and evaluation")

        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.data_processor.preprocess_data()

        # Save the preprocessor for later use
        self.data_processor.save_preprocessor()

        best_model_name = None
        best_f1_score = 0
        best_metrics = None

        # Train and evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            with mlflow.start_run(run_name=model_name):
                # Log model parameters
                mlflow.log_params(model.get_params())

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                if y_prob is not None:
                    roc_auc = roc_auc_score(y_test, y_prob)
                    mlflow.log_metric("roc_auc", roc_auc)

                # Store metrics for the best model
                current_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
                if y_prob is not None:
                    current_metrics["roc_auc"] = roc_auc

                # Perform cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
                mlflow.log_metric("cv_f1_mean", cv_scores.mean())
                mlflow.log_metric("cv_f1_std", cv_scores.std())

                # Create and save confusion matrix plot
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Not Poisonous", "Poisonous"],
                    yticklabels=["Not Poisonous", "Poisonous"],
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix - {model_name}")

                cm_plot_path = f"models/plots/{model_name}_confusion_matrix.png"
                plt.savefig(cm_plot_path)
                plt.close()

                # Log the plot as an artifact
                mlflow.log_artifact(cm_plot_path)

                # Save the model
                model_path = f"models/{model_name}.joblib"
                joblib.dump(model, model_path)

                # Log the model
                mlflow.sklearn.log_model(model, model_name)

                # Log feature importances if available
                if hasattr(model, "feature_importances_"):
                    feature_names = self.data_processor.get_feature_names()
                    importances = model.feature_importances_

                    # Create and save feature importance plot
                    plt.figure(figsize=(12, 8))
                    indices = np.argsort(importances)[-20:]  # Top 20 features
                    plt.barh(range(len(indices)), importances[indices], align="center")
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel("Feature Importance")
                    plt.title(f"Top 20 Feature Importances - {model_name}")

                    importance_plot_path = (
                        f"models/plots/{model_name}_feature_importance.png"
                    )
                    plt.savefig(importance_plot_path)
                    plt.close()

                    # Log the plot as an artifact
                    mlflow.log_artifact(importance_plot_path)

                # Print results
                logger.info(f"{model_name} Results:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                if y_prob is not None:
                    logger.info(f"  ROC AUC: {roc_auc:.4f}")
                logger.info(
                    f"  CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )

                # Track the best model
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_model_name = model_name
                    best_metrics = current_metrics

        # Register the best model
        self.register_best_model(best_model_name, best_metrics, X_test, y_test)

        return best_model_name

    def register_best_model(self, model_name, metrics, X_test, y_test):
        """
        Register the best model in MLflow and save metadata.

        Args:
            model_name (str): Name of the best model.
            metrics (dict): Dictionary containing model metrics.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
        """
        logger.info(f"Registering best model: {model_name}")

        # Load the best model
        model_path = f"models/{model_name}.joblib"
        best_model = joblib.load(model_path)

        # Create a production model directory
        prod_dir = "models/production"
        os.makedirs(prod_dir, exist_ok=True)

        # Save the model to the production directory
        prod_model_path = os.path.join(prod_dir, "model.joblib")
        joblib.dump(best_model, prod_model_path)

        # Save model metadata
        metadata = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "description": f"Best performing model ({model_name}) for mushroom classification",
        }

        with open(os.path.join(prod_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Best model registered and saved to {prod_model_path}")

        # Register the model in MLflow
        with mlflow.start_run(run_name=f"register_{model_name}"):
            mlflow.sklearn.log_model(
                best_model,
                "production_model",
                registered_model_name="mushroom_classifier",
            )

            # Log the metadata
            with open("models/production/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            mlflow.log_artifact("models/production/metadata.json")

            logger.info("Model registered in MLflow as 'mushroom_classifier'")


if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model = trainer.train_and_evaluate()
    logger.info(f"Training completed. Best model: {best_model}")
