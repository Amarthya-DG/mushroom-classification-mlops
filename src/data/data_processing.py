import logging
import os

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor.
        """
        self.categorical_features = [
            "cap_shape",
            "cap_surface",
            "cap_color",
            "odor",
            "gill_attachment",
            "gill_spacing",
            "gill_size",
            "gill_color",
            "stalk_shape",
            "stalk_surface_above_ring",
            "stalk_surface_belows_ring",
            "stalk_color_above_ring",
            "stalk_color_below_ring",
            "veil_type",
            "veil_color",
            "number_of_rings",
            "ring_type",
            "spore_print_color",
            "population",
            "habitat",
        ]
        self.boolean_features = ["has_bruises"]
        self.target = "is_poisonous"
        self.preprocessor = None
        self.label_encoder = None

    def load_data(self):
        """
        Load the mushroom dataset from Hugging Face.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        logger.info("Loading data from Hugging Face datasets")
        try:
            # Load the dataset
            dataset = load_dataset("mstz/mushroom")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset["train"])

            # The target variable is already binary (1 for poisonous, 0 for edible)
            logger.info(f"Data loaded successfully with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df=None, train=True):
        """
        Preprocess the data by encoding categorical features and splitting into train/test sets.

        Args:
            df (pd.DataFrame, optional): The dataset to preprocess. If None, loads the data.
            train (bool, optional): Whether to fit the preprocessor or just transform. Defaults to True.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) if train=True, else (X, y)
        """
        if df is None:
            df = self.load_data()

        # Convert boolean features to boolean type
        for col in self.boolean_features:
            df[col] = df[col].astype(bool)

        # Create feature matrix and target vector
        X = df.drop(columns=[self.target])
        y = df[self.target]

        if train:
            # Create and fit the preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        self.categorical_features,
                    ),
                ],
                remainder="passthrough",
            )

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Fit and transform the training data
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            logger.info(
                f"Data preprocessed successfully. Training set shape: {X_train_processed.shape}"
            )

            return X_train_processed, X_test_processed, y_train, y_test
        else:
            # Just transform the data using the fitted preprocessor
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessor not fitted. Call preprocess_data with train=True first."
                )

            X_processed = self.preprocessor.transform(X)

            logger.info(f"Data transformed successfully. Shape: {X_processed.shape}")

            return X_processed, y

    def save_preprocessor(self, output_path="models"):
        """
        Save the fitted preprocessor to disk.

        Args:
            output_path (str, optional): Directory to save the preprocessor. Defaults to 'models'.
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not fitted. Call preprocess_data with train=True first."
            )

        os.makedirs(output_path, exist_ok=True)
        preprocessor_path = os.path.join(output_path, "preprocessor.joblib")

        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

    def load_preprocessor(self, input_path="models"):
        """
        Load a fitted preprocessor from disk.

        Args:
            input_path (str, optional): Directory where the preprocessor is saved. Defaults to 'models'.
        """
        preprocessor_path = os.path.join(input_path, "preprocessor.joblib")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

    def get_feature_names(self):
        """
        Get the feature names after one-hot encoding.

        Returns:
            list: List of feature names.
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not fitted. Call preprocess_data with train=True first."
            )

        # Get feature names from one-hot encoder
        cat_features = self.preprocessor.transformers_[0][1].get_feature_names_out(
            self.categorical_features
        )

        # Combine with boolean features
        all_features = list(cat_features) + self.boolean_features

        return all_features

    def prepare_inference_input(self, input_data):
        """
        Prepare input data for inference.

        Args:
            input_data (dict): Input data as a dictionary.

        Returns:
            numpy.ndarray: Processed input data ready for model prediction.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Load a preprocessor first.")

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert boolean features
        for col in self.boolean_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(bool)

        # Transform the input data
        processed_input = self.preprocessor.transform(input_df)

        return processed_input


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    df = processor.load_data()

    print("Dataset info:")
    print(df.info())
    print("\nSample data:")
    print(df.head())

    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    processor.save_preprocessor()

    print("\nProcessed data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
