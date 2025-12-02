"""
IDS Classifier Model Training Script

This script trains a Decision Tree classifier on network intrusion detection data.
It includes data loading, preprocessing, model training, evaluation, and artifact saving.

Usage:
    python src/model/train.py
"""

import os
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import kagglehub

# Import preprocessing utilities
from preprocessing import log2_transform


# Configuration
RANDOM_SEED = 635458
TEST_SIZE = 0.3
VAL_SIZE = 0.3
MODEL_CRITERION = "entropy"
MODEL_MAX_DEPTH = 6
CV_FOLDS = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Download and load the IDS dataset from Kaggle."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("ayushparwal2026/intrusion-detection-classifier")
    data = pd.read_csv(os.path.join(path, "datacopy.csv"))
    print(f"Dataset loaded: {len(data)} records, {len(data.columns)} columns")
    return data


def identify_drop_columns(data):
    """
    Identify columns with very low diversity (>97% same value).

    Returns:
        List of column names to drop
    """
    drop_cols = []
    for col in data.columns:
        value_counts = data[col].value_counts()
        if len(value_counts) > 0:
            max_freq = value_counts.iloc[0] / len(data)
            if max_freq > 0.97:
                drop_cols.append(col)

    print(f"Identified {len(drop_cols)} low-diversity columns to drop: {drop_cols}")
    return drop_cols


def prepare_features(data):
    """
    Prepare feature sets and create target variable.

    Returns:
        Tuple of (data with is_attack, categorical_cols, count_cols, bytes_cols, drop_cols)
    """
    # Create binary target
    data["is_attack"] = data["Attack Type"].apply(lambda x: 1 if x != "normal" else 0)

    # Define feature categories
    categorical_cols = ["service", "flag", "logged_in", "protocol_type"]
    count_cols = ["count", "dst_host_count", "dst_host_srv_count", "srv_count"]
    bytes_cols = ["src_bytes", "dst_bytes"]
    rate_cols = [col for col in data.columns if "rate" in col]
    target_cols = ["is_attack", "target", "Attack Type"]

    # Identify and drop low diversity columns
    drop_cols = identify_drop_columns(data)
    data = data.drop(columns=drop_cols)

    return data, categorical_cols, count_cols, bytes_cols


def create_preprocessor(categorical_cols, count_cols, bytes_cols):
    """
    Create preprocessing pipeline.

    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Log2 transformation pipeline for byte columns
    bytes_size_transform = FunctionTransformer(
        func=log2_transform, feature_names_out="one-to-one", validate=True
    )

    bytes_size_pipeline = Pipeline(
        steps=[("log2", bytes_size_transform), ("minmax", MinMaxScaler())]
    )

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("minmax", MinMaxScaler(), count_cols),
            ("bytes_size", bytes_size_pipeline, bytes_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor


def identify_post_preprocessing_drops(x_train_df):
    """
    Identify columns to drop after preprocessing based on feature frequency.

    Returns:
        List of column names to drop
    """
    after_preprocess_drop_cols = ["service_other", "protocol_type_udp"]

    # Drop service columns with < 50 occurrences
    service_cols = [col for col in x_train_df.columns if "service" in col]
    service_counts = x_train_df[service_cols].sum()
    service_counts_small = service_counts[service_counts < 50]
    after_preprocess_drop_cols.extend(service_counts_small.index.tolist())

    # Drop flag columns with < 10 occurrences
    flag_cols = [col for col in x_train_df.columns if "flag" in col]
    flag_counts = x_train_df[flag_cols].sum()
    flag_counts_small = flag_counts[flag_counts < 10]
    after_preprocess_drop_cols.extend(flag_counts_small.index.tolist())

    print(f"Identified {len(after_preprocess_drop_cols)} post-preprocessing columns to drop")
    return after_preprocess_drop_cols


def preprocess_data(X, preprocessor, is_training=False):
    """
    Apply preprocessing to features.

    Args:
        X: Raw feature DataFrame
        preprocessor: Fitted ColumnTransformer
        is_training: Whether this is training data (for fitting)

    Returns:
        Preprocessed DataFrame with column names
    """
    if is_training:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    # Get feature names
    col_names = preprocessor.get_feature_names_out().tolist()
    col_names = [name.split("__")[-1] for name in col_names]

    # Convert to DataFrame
    X_df = pd.DataFrame(X_transformed, columns=col_names)

    return X_df


def train_model(x_train, y_train, criterion=MODEL_CRITERION, max_depth=MODEL_MAX_DEPTH):
    """
    Train Decision Tree classifier.

    Returns:
        Trained model
    """
    print(f"\nTraining Decision Tree (criterion={criterion}, max_depth={max_depth})...")
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    print("Training complete!")
    return model


def evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate model performance on train, validation, and test sets.

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = {}

    # Cross-validation on training set
    print("\nPerforming 10-fold cross-validation on training set...")
    cv_results = cross_validate(
        model,
        x_train,
        y_train,
        cv=CV_FOLDS,
        scoring=["accuracy", "precision", "recall", "f1"],
    )

    results["cv"] = {
        "accuracy": (cv_results["test_accuracy"].mean(), cv_results["test_accuracy"].std()),
        "precision": (cv_results["test_precision"].mean(), cv_results["test_precision"].std()),
        "recall": (cv_results["test_recall"].mean(), cv_results["test_recall"].std()),
        "f1": (cv_results["test_f1"].mean(), cv_results["test_f1"].std()),
    }

    print("\nCross-Validation Results (Mean ± Std):")
    for metric, (mean, std) in results["cv"].items():
        print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")

    # Validation set evaluation
    print("\n" + "-" * 60)
    print("Validation Set Performance:")
    print("-" * 60)
    y_val_pred = model.predict(x_val)

    results["val"] = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "f1": f1_score(y_val, y_val_pred),
    }

    for metric, score in results["val"].items():
        print(f"  {metric.capitalize():12s}: {score:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Normal", "Attack"]))

    # Test set evaluation
    print("-" * 60)
    print("Test Set Performance:")
    print("-" * 60)
    y_test_pred = model.predict(x_test)

    results["test"] = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
    }

    for metric, score in results["test"].items():
        print(f"  {metric.capitalize():12s}: {score:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Normal", "Attack"]))

    # Feature importance
    print("\n" + "-" * 60)
    print("Top 10 Most Important Features:")
    print("-" * 60)
    feature_importances = model.feature_importances_
    feature_names = x_train.columns
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:35s}: {row['Importance']:.6f}")

    results["feature_importance"] = importance_df

    return results


def save_artifacts(model, preprocessor, drop_cols, results):
    """
    Save model, preprocessor, and metadata to disk.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")

    # Save post-preprocessing drop columns
    drop_cols_path = MODELS_DIR / "drop_columns.joblib"
    joblib.dump(drop_cols, drop_cols_path)
    print(f"Drop columns list saved to: {drop_cols_path}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": "DecisionTreeClassifier",
        "hyperparameters": {
            "criterion": MODEL_CRITERION,
            "max_depth": MODEL_MAX_DEPTH,
            "random_state": RANDOM_SEED,
        },
        "test_metrics": results["test"],
        "val_metrics": results["val"],
        "cv_metrics": {k: v[0] for k, v in results["cv"].items()},  # mean values
    }

    metadata_path = MODELS_DIR / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")

    # Save feature importance as CSV
    importance_path = MODELS_DIR / "feature_importance.csv"
    results["feature_importance"].to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

    print("\n" + "=" * 60)
    print("All artifacts saved successfully!")
    print("=" * 60)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("IDS CLASSIFIER TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    data = load_data()

    # Prepare features
    data, categorical_cols, count_cols, bytes_cols = prepare_features(data)

    # Split data (stratified)
    print(f"\nSplitting data (test_size={TEST_SIZE}, val_size={VAL_SIZE})...")
    target_cols = ["is_attack", "target", "Attack Type"]
    X = data.drop(columns=target_cols)
    y = data["is_attack"]

    temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        temp, y_temp, test_size=TEST_SIZE, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")

    # Create and fit preprocessor
    print("\nCreating preprocessing pipeline...")
    preprocessor = create_preprocessor(categorical_cols, count_cols, bytes_cols)

    # Preprocess data
    print("Preprocessing training data...")
    X_train_processed = preprocess_data(X_train, preprocessor, is_training=True)
    print("Preprocessing validation data...")
    X_val_processed = preprocess_data(X_val, preprocessor, is_training=False)
    print("Preprocessing test data...")
    X_test_processed = preprocess_data(X_test, preprocessor, is_training=False)

    print(f"Features after preprocessing: {X_train_processed.shape[1]}")

    # Identify and drop low-frequency columns
    drop_cols = identify_post_preprocessing_drops(X_train_processed)
    X_train_processed = X_train_processed.drop(columns=drop_cols)
    X_val_processed = X_val_processed.drop(columns=drop_cols)
    X_test_processed = X_test_processed.drop(columns=drop_cols)

    print(f"Features after dropping low-frequency columns: {X_train_processed.shape[1]}")

    # Reset indices for target variables
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Train model
    model = train_model(X_train_processed, y_train)

    # Evaluate model
    results = evaluate_model(
        model, X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test
    )

    # Save artifacts
    save_artifacts(model, preprocessor, drop_cols, results)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
