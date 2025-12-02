"""
IDS Classifier Inference Module

This module provides inference capabilities for the trained IDS classifier.
It loads the trained model and preprocessor, and provides methods for making predictions.

Usage:
    from src.model.inference import IDSClassifier

    # Initialize classifier
    classifier = IDSClassifier()

    # Predict on a single sample
    result = classifier.predict_one(sample_dict)

    # Predict on multiple samples
    results = classifier.predict_batch(samples_df)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, List, Union
import warnings

warnings.filterwarnings("ignore")


class IDSClassifier:
    """
    IDS Classifier for network intrusion detection.

    This class loads a trained model and provides methods for inference.
    """

    def __init__(self, models_dir: Union[str, Path] = None):
        """
        Initialize the IDS Classifier.

        Args:
            models_dir: Path to directory containing model artifacts.
                       If None, uses default 'models/' directory.
        """
        if models_dir is None:
            # Default to models/ directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)

        self.models_dir = models_dir

        # Load artifacts
        self._load_artifacts()

        print(f"IDS Classifier loaded successfully!")
        print(f"Model: {self.metadata['model_type']}")
        print(f"Test F1 Score: {self.metadata['test_metrics']['f1']:.4f}")

    def _load_artifacts(self):
        """Load model, preprocessor, and metadata from disk."""
        # Load model
        model_path = self.models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please run train.py first to generate model artifacts."
            )
        self.model = joblib.load(model_path)

        # Load preprocessor
        preprocessor_path = self.models_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        # Load drop columns list
        drop_cols_path = self.models_dir / "drop_columns.joblib"
        if not drop_cols_path.exists():
            raise FileNotFoundError(f"Drop columns file not found at {drop_cols_path}")
        self.drop_cols = joblib.load(drop_cols_path)

        # Load metadata
        metadata_path = self.models_dir / "model_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        self.metadata = joblib.load(metadata_path)

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw features using the trained preprocessor.

        Args:
            X: Raw features DataFrame

        Returns:
            Preprocessed features DataFrame
        """
        # Apply column transformer
        X_transformed = self.preprocessor.transform(X)

        # Get feature names
        col_names = self.preprocessor.get_feature_names_out().tolist()
        col_names = [name.split("__")[-1] for name in col_names]

        # Convert to DataFrame
        X_df = pd.DataFrame(X_transformed, columns=col_names)

        # Drop low-frequency columns
        X_df = X_df.drop(columns=self.drop_cols, errors="ignore")

        return X_df

    def predict_one(self, sample: Dict) -> Dict:
        """
        Predict on a single sample.

        Args:
            sample: Dictionary containing feature values

        Returns:
            Dictionary with prediction results:
                - prediction: 'normal' or 'attack'
                - prediction_label: 0 or 1
                - confidence: prediction confidence score
                - probabilities: class probabilities [prob_normal, prob_attack]
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample])

        # Get prediction
        X_processed = self._preprocess(df)
        pred_label = self.model.predict(X_processed)[0]
        pred_proba = self.model.predict_proba(X_processed)[0] if hasattr(self.model, 'predict_proba') else None

        # Prepare result
        result = {
            "prediction": "attack" if pred_label == 1 else "normal",
            "prediction_label": int(pred_label),
        }

        if pred_proba is not None:
            result["confidence"] = float(pred_proba[pred_label])
            result["probabilities"] = {
                "normal": float(pred_proba[0]),
                "attack": float(pred_proba[1])
            }

        return result

    def predict_batch(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on multiple samples.

        Args:
            samples: DataFrame containing feature values for multiple samples

        Returns:
            DataFrame with original data plus prediction columns:
                - prediction: 'normal' or 'attack'
                - prediction_label: 0 or 1
                - confidence: prediction confidence score
                - prob_normal: probability of normal class
                - prob_attack: probability of attack class
        """
        # Get predictions
        X_processed = self._preprocess(samples)
        pred_labels = self.model.predict(X_processed)

        # Create results DataFrame
        results = samples.copy()
        results["prediction_label"] = pred_labels
        results["prediction"] = results["prediction_label"].apply(
            lambda x: "attack" if x == 1 else "normal"
        )

        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(X_processed)
            results["prob_normal"] = pred_proba[:, 0]
            results["prob_attack"] = pred_proba[:, 1]
            results["confidence"] = [
                pred_proba[i, pred_labels[i]] for i in range(len(pred_labels))
            ]

        return results

    def predict_raw(self, samples: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """
        Convenience method that accepts either a single sample dict or DataFrame.

        Args:
            samples: Either a dictionary (single sample) or DataFrame (batch)

        Returns:
            Dictionary for single sample, DataFrame for batch
        """
        if isinstance(samples, dict):
            return self.predict_one(samples)
        elif isinstance(samples, pd.DataFrame):
            return self.predict_batch(samples)
        else:
            raise TypeError("Input must be either a dictionary or pandas DataFrame")

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_type": self.metadata["model_type"],
            "hyperparameters": self.metadata["hyperparameters"],
            "test_metrics": self.metadata["test_metrics"],
            "timestamp": self.metadata["timestamp"],
        }


def main():
    """
    Example usage of the IDSClassifier.
    """
    print("=" * 60)
    print("IDS CLASSIFIER INFERENCE DEMO")
    print("=" * 60)

    try:
        # Initialize classifier
        classifier = IDSClassifier()

        # Display model info
        print("\n" + "-" * 60)
        print("Model Information:")
        print("-" * 60)
        info = classifier.get_model_info()
        print(f"Model Type: {info['model_type']}")
        print(f"Training Date: {info['timestamp']}")
        print(f"\nTest Set Performance:")
        for metric, value in info['test_metrics'].items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")

        # Example: Create a dummy sample for testing
        print("\n" + "-" * 60)
        print("Example Prediction (Dummy Data):")
        print("-" * 60)

        dummy_sample = {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 181,
            "dst_bytes": 5450,
            "logged_in": 1,
            "count": 8,
            "srv_count": 8,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 9,
            "dst_host_srv_count": 9,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.11,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        }

        result = classifier.predict_one(dummy_sample)

        print(f"Prediction: {result['prediction']}")
        print(f"Label: {result['prediction_label']}")
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.4f}")
        if 'probabilities' in result:
            print(f"Probabilities:")
            print(f"  Normal: {result['probabilities']['normal']:.4f}")
            print(f"  Attack: {result['probabilities']['attack']:.4f}")

        print("\n" + "=" * 60)
        print("Inference demo completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python src/model/train.py' first to train the model.")


if __name__ == "__main__":
    main()
