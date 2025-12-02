"""
Model Artifacts Verification Script

This script verifies that all required model artifacts exist, can be loaded,
and are working correctly. Run this after training to ensure the model is
ready for deployment.

Usage:
    python src/model/verify_artifacts.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import preprocessing utilities (required for loading serialized preprocessor)
from src.model.preprocessing import log2_transform  # noqa: F401
from src.model.inference import IDSClassifier


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"  ✓ {description:30s} [{size_kb:>8.1f} KB]")
        return True
    else:
        print(f"  ✗ {description:30s} [MISSING]")
        return False


def verify_artifacts():
    """Verify all model artifacts exist and can be loaded."""
    print("=" * 70)
    print("MODEL ARTIFACTS VERIFICATION")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models"

    # Check if models directory exists
    if not models_dir.exists():
        print(f"\n✗ Models directory not found: {models_dir}")
        print("\nPlease run 'python src/model/train.py' first to train the model.")
        return False

    print(f"\nModels directory: {models_dir}")
    print("\n" + "-" * 70)
    print("Checking Required Artifacts:")
    print("-" * 70)

    # Required artifacts
    required_files = {
        "model.joblib": "Trained classifier",
        "preprocessor.joblib": "Feature preprocessor",
        "drop_columns.joblib": "Columns to drop",
        "model_metadata.joblib": "Model metadata",
    }

    all_exist = True
    for filename, description in required_files.items():
        file_path = models_dir / filename
        if not check_file_exists(file_path, description):
            all_exist = False

    # Optional artifacts
    print("\n" + "-" * 70)
    print("Optional Artifacts:")
    print("-" * 70)

    optional_files = {
        "feature_importance.csv": "Feature importance scores",
    }

    for filename, description in optional_files.items():
        check_file_exists(models_dir / filename, description)

    if not all_exist:
        print("\n" + "=" * 70)
        print("✗ VERIFICATION FAILED: Missing required artifacts")
        print("=" * 70)
        print("\nPlease run: python src/model/train.py")
        return False

    # Try loading artifacts
    print("\n" + "-" * 70)
    print("Loading Artifacts:")
    print("-" * 70)

    try:
        # Load model
        print("  Loading trained model...")
        model = joblib.load(models_dir / "model.joblib")
        print(f"    ✓ Model type: {type(model).__name__}")

        # Load preprocessor
        print("  Loading preprocessor...")
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
        print(f"    ✓ Preprocessor loaded successfully")

        # Load drop columns
        print("  Loading drop columns...")
        drop_cols = joblib.load(models_dir / "drop_columns.joblib")
        print(f"    ✓ {len(drop_cols)} columns to drop")

        # Load metadata
        print("  Loading metadata...")
        metadata = joblib.load(models_dir / "model_metadata.joblib")
        print(f"    ✓ Metadata loaded successfully")

    except Exception as e:
        print(f"\n  ✗ Error loading artifacts: {e}")
        return False

    # Display metadata
    print("\n" + "-" * 70)
    print("Model Metadata:")
    print("-" * 70)
    print(f"  Model Type:       {metadata['model_type']}")
    print(f"  Training Date:    {metadata['timestamp']}")
    print(f"\n  Hyperparameters:")
    for param, value in metadata['hyperparameters'].items():
        print(f"    {param:20s}: {value}")

    print(f"\n  Test Set Performance:")
    for metric, value in metadata['test_metrics'].items():
        print(f"    {metric.capitalize():20s}: {value:.4f}")

    # Test inference
    print("\n" + "-" * 70)
    print("Testing Inference:")
    print("-" * 70)

    try:
        # Initialize classifier
        print("  Initializing IDSClassifier...")
        classifier = IDSClassifier(models_dir=models_dir)
        print("    ✓ Classifier initialized")

        # Create a dummy sample
        dummy_sample = {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 181,
            "dst_bytes": 5450,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
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

        # Test prediction
        print("  Running test prediction...")
        result = classifier.predict_one(dummy_sample)

        print("    ✓ Prediction successful")
        print(f"\n  Test Prediction Results:")
        print(f"    Prediction:   {result['prediction']}")
        print(f"    Label:        {result['prediction_label']}")
        if 'confidence' in result:
            print(f"    Confidence:   {result['confidence']:.4f}")
        if 'probabilities' in result:
            print(f"    Prob(Normal): {result['probabilities']['normal']:.4f}")
            print(f"    Prob(Attack): {result['probabilities']['attack']:.4f}")

    except Exception as e:
        print(f"\n  ✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All checks passed
    print("\n" + "=" * 70)
    print("✓ VERIFICATION SUCCESSFUL")
    print("=" * 70)
    print("\nAll model artifacts are present and working correctly.")
    print("The model is ready for deployment!")

    return True


def main():
    """Main verification routine."""
    success = verify_artifacts()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
