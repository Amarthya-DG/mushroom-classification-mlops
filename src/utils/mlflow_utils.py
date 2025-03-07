import json
import logging
import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLflowManager:
    def __init__(self, tracking_uri=None, experiment_name="mushroom-classification"):
        """
        Initialize the MLflow manager.

        Args:
            tracking_uri (str, optional): MLflow tracking URI. Defaults to None (local).
            experiment_name (str, optional): Name of the MLflow experiment. Defaults to "mushroom-classification".
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        self.experiment_name = experiment_name

        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
            logger.info(
                f"Using experiment '{experiment_name}' (ID: {self.experiment.experiment_id})"
            )
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def log_model_to_registry(self, model, model_name, run_id=None):
        """
        Log a model to the MLflow model registry.

        Args:
            model: The trained model to log.
            model_name (str): Name for the registered model.
            run_id (str, optional): ID of an existing run to log to. Defaults to None (creates new run).

        Returns:
            str: The model version.
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    result = mlflow.sklearn.log_model(
                        model, "model", registered_model_name=model_name
                    )
            else:
                with mlflow.start_run(experiment_id=self.experiment.experiment_id):
                    result = mlflow.sklearn.log_model(
                        model, "model", registered_model_name=model_name
                    )

            logger.info(f"Model '{model_name}' logged to MLflow model registry")
            return result.model_uri
        except Exception as e:
            logger.error(f"Error logging model to registry: {e}")
            raise

    def get_latest_model_version(self, model_name):
        """
        Get the latest version of a registered model.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            int: Latest version number.
        """
        try:
            latest_version = 1
            for mv in self.client.search_model_versions(f"name='{model_name}'"):
                version = int(mv.version)
                if version > latest_version:
                    latest_version = version
            return latest_version
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return 1

    def transition_model_stage(self, model_name, version, stage):
        """
        Transition a model version to a different stage.

        Args:
            model_name (str): Name of the registered model.
            version (int): Version number.
            stage (str): Target stage ('Staging', 'Production', or 'Archived').

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(
                f"Model '{model_name}' version {version} transitioned to {stage}"
            )
            return True
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False

    def get_production_model(self, model_name):
        """
        Get the production version of a model.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            tuple: (model_uri, version) or (None, None) if no production model exists.
        """
        try:
            for mv in self.client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    model_uri = f"models:/{model_name}/{mv.version}"
                    return model_uri, mv.version
            return None, None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None, None

    def log_model_metadata(self, run_id, metadata):
        """
        Log model metadata to an MLflow run.

        Args:
            run_id (str): ID of the run to log to.
            metadata (dict): Metadata to log.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create a temporary file with the metadata
            metadata_file = "temp_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Log the file as an artifact
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(metadata_file)

            # Clean up
            os.remove(metadata_file)

            logger.info(f"Model metadata logged to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging model metadata: {e}")
            return False

    def create_deployment_event(self, model_name, version, environment):
        """
        Log a deployment event.

        Args:
            model_name (str): Name of the deployed model.
            version (int): Version of the deployed model.
            environment (str): Environment where the model was deployed.

        Returns:
            str: ID of the created run.
        """
        try:
            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=f"deployment-{model_name}-v{version}",
            ) as run:
                # Log deployment information
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_version", version)
                mlflow.log_param("environment", environment)
                mlflow.log_param("deployment_time", datetime.now().isoformat())

                # Create a deployment event file
                event = {
                    "model_name": model_name,
                    "model_version": version,
                    "environment": environment,
                    "deployment_time": datetime.now().isoformat(),
                    "deployed_by": os.environ.get("USER", "unknown"),
                }

                event_file = "deployment_event.json"
                with open(event_file, "w") as f:
                    json.dump(event, f, indent=2)

                # Log the file as an artifact
                mlflow.log_artifact(event_file)

                # Clean up
                os.remove(event_file)

                logger.info(
                    f"Deployment event logged for {model_name} v{version} to {environment}"
                )
                return run.info.run_id
        except Exception as e:
            logger.error(f"Error logging deployment event: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    mlflow_manager = MLflowManager()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment name: {mlflow_manager.experiment_name}")
    print(f"Experiment ID: {mlflow_manager.experiment.experiment_id}")
