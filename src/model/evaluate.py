"""
Model Evaluation Script

This script loads a trained model and evaluates it on test data, generating
a comprehensive evaluation report with metrics, confusion matrix, and
classification report.

Usage:
    python src/model/evaluate.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import kagglehub

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import preprocessing utilities (required for loading serialized preprocessor)
from src.model.preprocessing import log2_transform  # noqa: F401
from src.model.inference import IDSClassifier

# Configuration
RANDOM_SEED = 635458
MODELS_DIR = PROJECT_ROOT / "models"


def load_test_data():
    """
    Load and prepare test data.

    This replicates the exact same test split used during training.
    """
    print("Loading dataset from Kaggle...")
    path = kagglehub.dataset_download("ayushparwal2026/intrusion-detection-classifier")
    data = pd.read_csv(path + '/datacopy.csv')

    print(f"Dataset loaded: {len(data)} records")

    # Create target variable
    data['is_attack'] = (data['Attack Type'] != 'normal').astype(int)

    # Replicate the training split to get the exact test set
    from sklearn.model_selection import train_test_split

    # Define target and feature columns
    target_cols = ["is_attack", "target", "Attack Type"]
    X = data.drop(columns=target_cols)
    y = data['is_attack']

    # First split: 70% temp, 30% val
    temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    # Second split: 70% train, 30% test (of the 70% temp)
    X_train, X_test, y_train, y_test = train_test_split(
        temp, y_temp, test_size=0.30, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"Test set: {len(X_test)} records")
    print(f"  - Normal: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
    print(f"  - Attack: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

    return X_test, y_test


def evaluate_model(classifier: IDSClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model on test data.

    Args:
        classifier: Trained IDSClassifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 70)

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = []
    y_proba = []

    for idx, row in X_test.iterrows():
        sample = row.to_dict()
        result = classifier.predict_one(sample)
        y_pred.append(result['prediction_label'])

        if 'probabilities' in result:
            y_proba.append([
                result['probabilities']['normal'],
                result['probabilities']['attack']
            ])

    y_pred = np.array(y_pred)
    y_test_array = y_test.values

    # Calculate metrics
    print("\n" + "-" * 70)
    print("Overall Metrics:")
    print("-" * 70)

    accuracy = accuracy_score(y_test_array, y_pred)
    precision = precision_score(y_test_array, y_pred)
    recall = recall_score(y_test_array, y_pred)
    f1 = f1_score(y_test_array, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")

    # Confusion matrix
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)

    cm = confusion_matrix(y_test_array, y_pred)

    print("\n              Predicted")
    print("               Normal  Attack")
    print(f"Actual Normal  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Attack  {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Per-class metrics
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    print()
    print(classification_report(
        y_test_array,
        y_pred,
        target_names=["Normal", "Attack"],
        digits=4
    ))

    # Error analysis
    print("-" * 70)
    print("Error Analysis:")
    print("-" * 70)

    false_positives = np.sum((y_test_array == 0) & (y_pred == 1))
    false_negatives = np.sum((y_test_array == 1) & (y_pred == 0))
    true_positives = np.sum((y_test_array == 1) & (y_pred == 1))
    true_negatives = np.sum((y_test_array == 0) & (y_pred == 0))

    print(f"  True Positives:  {true_positives:6d} (Correctly identified attacks)")
    print(f"  True Negatives:  {true_negatives:6d} (Correctly identified normal)")
    print(f"  False Positives: {false_positives:6d} (Normal flagged as attack)")
    print(f"  False Negatives: {false_negatives:6d} (Attack missed)")

    if false_positives > 0:
        fp_rate = false_positives / (false_positives + true_negatives)
        print(f"\n  False Positive Rate: {fp_rate:.4f} ({fp_rate * 100:.2f}%)")

    if false_negatives > 0:
        fn_rate = false_negatives / (false_negatives + true_positives)
        print(f"  False Negative Rate: {fn_rate:.4f} ({fn_rate * 100:.2f}%)")

    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'errors': {
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    }

    return results


def save_evaluation_report(results, classifier):
    """Save evaluation results to a report file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = MODELS_DIR / f"evaluation_report_{timestamp}.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Model info
        info = classifier.get_model_info()
        f.write("Model Information:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model Type:     {info['model_type']}\n")
        f.write(f"Training Date:  {info['timestamp']}\n\n")

        f.write("Hyperparameters:\n")
        for param, value in info['hyperparameters'].items():
            f.write(f"  {param:20s}: {value}\n")

        # Metrics
        f.write("\n" + "-" * 70 + "\n")
        f.write("Test Set Metrics:\n")
        f.write("-" * 70 + "\n")
        for metric, value in results['metrics'].items():
            f.write(f"  {metric.capitalize():12s}: {value:.4f}\n")

        # Confusion matrix
        cm = results['confusion_matrix']
        f.write("\n" + "-" * 70 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 70 + "\n")
        f.write("\n              Predicted\n")
        f.write("               Normal  Attack\n")
        f.write(f"Actual Normal  {cm[0][0]:6d}  {cm[0][1]:6d}\n")
        f.write(f"       Attack  {cm[1][0]:6d}  {cm[1][1]:6d}\n")

        # Errors
        f.write("\n" + "-" * 70 + "\n")
        f.write("Error Analysis:\n")
        f.write("-" * 70 + "\n")
        errors = results['errors']
        f.write(f"  True Positives:  {errors['true_positives']:6d}\n")
        f.write(f"  True Negatives:  {errors['true_negatives']:6d}\n")
        f.write(f"  False Positives: {errors['false_positives']:6d}\n")
        f.write(f"  False Negatives: {errors['false_negatives']:6d}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Report generated: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"\nEvaluation report saved to: {report_path}")


def main():
    """Main evaluation routine."""
    print("=" * 70)
    print("IDS CLASSIFIER EVALUATION")
    print("=" * 70)

    # Load classifier
    print("\nLoading trained classifier...")
    try:
        classifier = IDSClassifier(models_dir=MODELS_DIR)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python src/model/train.py' first to train the model.")
        sys.exit(1)

    # Load test data
    X_test, y_test = load_test_data()

    # Evaluate
    results = evaluate_model(classifier, X_test, y_test)

    # Save report
    save_evaluation_report(results, classifier)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
    print(f"  Precision: {results['metrics']['precision']:.4f}")
    print(f"  Recall:    {results['metrics']['recall']:.4f}")
    print(f"  F1 Score:  {results['metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
