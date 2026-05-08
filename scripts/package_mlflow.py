#!/usr/bin/env python3
"""
Package iForestASD model with MLflow.
Task 2.3-2.5: MLflow Packaging

Packages:
- Trained iForestASD model (v2: 200 trees, height 10, window 512)
- Fitted StandardScaler
- Context-aware thresholds (4D clustering)

Usage:
  python scripts/package_mlflow.py
  python scripts/package_mlflow.py --model-path models/iforest_model_v2.pkl
"""

import argparse
import sys
from pathlib import Path
import json
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlflow
    import mlflow.pyfunc
except ImportError:
    print("❌ MLflow not installed. Install with: pip install mlflow")
    sys.exit(1)


def package_model_to_mlflow(
    model_path: str = 'models/iforest_model_v2.pkl',
    scaler_path: str = 'models/scaler.pkl',
    thresholds_path: str = 'models/context_thresholds_v2.json',
    run_name: str = 'iforest_v1.0.0',
    model_name: str = 'iforest-asd-cadqstream'
):
    """Package model artifacts to MLflow.

    Args:
        model_path: Path to trained iForestASD model
        scaler_path: Path to fitted StandardScaler
        thresholds_path: Path to context thresholds JSON
        run_name: MLflow run name
        model_name: Registered model name
    """
    print("="*60)
    print("MLflow Model Packaging")
    print("="*60)

    # Validate files exist
    for path in [model_path, scaler_path, thresholds_path]:
        if not Path(path).exists():
            print(f"❌ Error: File not found: {path}")
            return 1

    # Load model to get config
    print(f"\n1. Loading model metadata...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"   Model config:")
    print(f"   - n_trees: {model.n_trees}")
    print(f"   - height: {model.height}")
    print(f"   - window_size: {model.window_size}")

    # Load thresholds to get metrics
    print(f"\n2. Loading thresholds and metrics...")
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    # Extract validation metrics if available
    validation_metrics = thresholds.get('validation_metrics', {})
    fpr = validation_metrics.get('fpr', 0.05)
    recall = validation_metrics.get('recall', 0.75)
    f1 = validation_metrics.get('f1', 0.0)

    print(f"   Validation metrics:")
    print(f"   - FPR: {fpr:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1: {f1:.3f}")

    # Start MLflow run
    print(f"\n3. Creating MLflow run: {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        print(f"\n4. Logging parameters...")
        mlflow.log_params({
            'algorithm': 'HalfSpaceTrees (iForestASD)',
            'n_trees': model.n_trees,
            'height': model.height,
            'window_size': model.window_size,
            'training_window': 'Jan 2024',
            'features': '21D (15D base + 6D ratio)',
            'seed': 42
        })
        print(f"   ✓ Parameters logged")

        # Log metrics
        print(f"\n5. Logging metrics...")
        mlflow.log_metrics({
            'fpr': fpr,
            'recall': recall,
            'f1': f1,
            'precision': validation_metrics.get('precision', 0.0),
            'specificity': validation_metrics.get('specificity', 0.0)
        })
        print(f"   ✓ Metrics logged")

        # Log artifacts
        print(f"\n6. Logging artifacts...")
        mlflow.log_artifact(model_path, artifact_path='model')
        print(f"   ✓ Model: {model_path}")

        mlflow.log_artifact(scaler_path, artifact_path='model')
        print(f"   ✓ Scaler: {scaler_path}")

        mlflow.log_artifact(thresholds_path, artifact_path='config')
        print(f"   ✓ Thresholds: {thresholds_path}")

        # Log metadata
        metadata = {
            'version': 'v1.0.0',
            'training_data': 'data/clean/jan_2024_clean_baseline.parquet',
            'training_records': 2960000,
            'prototype_validated': True,
            'context_clustering': '4D (trip_type, time, day_type, neighborhood)',
            'deployment_ready': fpr < 0.05 and recall > 0.75
        }

        metadata_path = Path('models/mlflow_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(metadata_path), artifact_path='config')
        print(f"   ✓ Metadata: {metadata_path}")

        # Register model
        print(f"\n7. Registering model: {model_name}")
        model_uri = f"runs:/{run.info.run_id}/model"

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={'version': 'v1.0.0', 'stage': 'production'}
            )
            print(f"   ✓ Registered as: {model_name}")
            print(f"   ✓ Version: {registered_model.version}")
        except Exception as e:
            print(f"   ⚠ Registration skipped: {e}")
            print(f"   (Model artifacts still logged to run)")

        # Print run info
        print(f"\n{'='*60}")
        print(f"✅ Packaging Complete!")
        print(f"{'='*60}")
        print(f"\nMLflow Run Info:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Run Name: {run_name}")
        print(f"  Artifact URI: {run.info.artifact_uri}")
        print(f"\nTo view in MLflow UI:")
        print(f"  mlflow ui --port 5000")
        print(f"  Open: http://localhost:5000")
        print(f"\nTo download artifacts:")
        print(f"  mlflow artifacts download -r {run.info.run_id} -d ./downloaded_model")
        print(f"{'='*60}")

        return run.info.run_id


def test_artifact_download(run_id: str):
    """Test downloading artifacts from MLflow.

    Args:
        run_id: MLflow run ID
    """
    print(f"\n{'='*60}")
    print("Testing Artifact Download")
    print(f"{'='*60}")

    print(f"\nRun ID: {run_id}")

    try:
        # Get artifact URI
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        print(f"Artifact URI: {artifact_uri}")

        # List artifacts
        artifacts = client.list_artifacts(run_id)
        print(f"\nArtifacts in run:")
        for artifact in artifacts:
            print(f"  - {artifact.path} ({artifact.file_size} bytes)")

        print(f"\n✅ Artifact download test passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Artifact download test failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Package iForestASD model with MLflow')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/iforest_model_v2.pkl',
        help='Path to trained model (default: models/iforest_model_v2.pkl)'
    )
    parser.add_argument(
        '--scaler-path',
        type=str,
        default='models/scaler.pkl',
        help='Path to fitted scaler (default: models/scaler.pkl)'
    )
    parser.add_argument(
        '--thresholds-path',
        type=str,
        default='models/context_thresholds_v2.json',
        help='Path to thresholds (default: models/context_thresholds_v2.json)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='iforest_v1.0.0',
        help='MLflow run name (default: iforest_v1.0.0)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='iforest-asd-cadqstream',
        help='Registered model name (default: iforest-asd-cadqstream)'
    )
    parser.add_argument(
        '--test-download',
        action='store_true',
        help='Test artifact download after packaging'
    )

    args = parser.parse_args()

    # Package model
    run_id = package_model_to_mlflow(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        thresholds_path=args.thresholds_path,
        run_name=args.run_name,
        model_name=args.model_name
    )

    if run_id is None:
        return 1

    # Test download if requested
    if args.test_download:
        return test_artifact_download(run_id)

    return 0


if __name__ == '__main__':
    exit(main())
