"""
Offline Warmup Script for MemStream.

This script trains the MemStream autoencoder and initializes the Memory Module
using clean baseline data. The resulting checkpoint files are then loaded by
the Flink operator at runtime.

Usage:
    python scripts/warmup_memstream.py --data data/clean_baseline.parquet --output /models/memstream

Checkpoint files created:
    - memstream_weights.pt: Trained autoencoder weights
    - memstream_scaler.pkl: Normalization statistics (mean, std)
    - memstream_memory.pt: Initial memory state
    - memstream_config.json: Configuration metadata
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from src.features.vectorizer_25d import FeatureVectorizer25D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger('warmup')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Offline MemStream warmup: train AE and initialize memory'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to clean baseline data (CSV or Parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/models/memstream',
        help='Output directory for checkpoint files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Number of training epochs (default: 500)'
    )
    parser.add_argument(
        '--memory-len',
        type=int,
        default=50000,
        help='Memory module length (default: 50000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size (default: 256)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--in-dim',
        type=int,
        default=25,
        help='Input feature dimension (default: 25)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=50,
        help='Hidden layer dimension (default: 50)'
    )
    return parser.parse_args()


def load_data(data_path: str, vectorizer: FeatureVectorizer25D) -> np.ndarray:
    """
    Load baseline data and extract features.

    Args:
        data_path: Path to CSV or Parquet file
        vectorizer: FeatureVectorizer25D instance

    Returns:
        Feature matrix of shape (N, 25)
    """
    LOGGER.info(f"Loading data from {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    LOGGER.info(f"Loaded {len(df)} records")

    # Extract features
    records = df.to_dict('records')
    features = vectorizer.transform_batch(records)

    LOGGER.info(f"Extracted features shape: {features.shape}")

    # Handle any NaN or inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def train_memstream(
    features: np.ndarray,
    args: argparse.Namespace
) -> tuple[MemStreamCore, MemStreamConfig]:
    """
    Train MemStream on baseline data.

    Args:
        features: Feature matrix [N, 25]
        args: Command-line arguments

    Returns:
        Tuple of (trained MemStreamCore, config used)
    """
    LOGGER.info("Initializing MemStream config")

    # Configure MemStream
    cfg = MemStreamConfig()
    cfg.in_dim = args.in_dim
    cfg.hidden_dim = args.hidden_dim
    cfg.out_dim = args.in_dim
    cfg.memory_len = args.memory_len
    cfg.warmup_epochs = args.epochs
    cfg.warmup_batch_size = args.batch_size
    cfg.warmup_noise_std = 0.1
    cfg.default_beta = 0.5
    cfg.k_neighbors = 10
    cfg.seed = args.seed

    # Set determinism
    set_determinism(args.seed)

    # Initialize MemStream
    ms_core = MemStreamCore(cfg=cfg, device='cpu')
    LOGGER.info("MemStreamCore initialized")

    # Train (warmup)
    LOGGER.info(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    ms_core.warmup(features, verbose=True)

    return ms_core, cfg


def save_checkpoint(
    ms_core: MemStreamCore,
    cfg: MemStreamConfig,
    args: argparse.Namespace
):
    """
    Save MemStream checkpoint files.

    Files created:
    - memstream_weights.pt: AE state + normalization stats
    - memstream_scaler.pkl: mean and std tensors for inference
    - memstream_memory.pt: Memory module state
    - memstream_config.json: Configuration metadata
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save AE weights + normalization stats
    weights_path = output_dir / 'memstream_weights.pt'
    torch.save({
        'ae_state_dict': ms_core.ae.state_dict(),
        'mean': ms_core.mean.cpu(),
        'std': ms_core.std.cpu(),
        'cfg_in_dim': cfg.in_dim,
        'cfg_hidden_dim': cfg.hidden_dim,
    }, weights_path)
    LOGGER.info(f"Saved weights to {weights_path}")

    # 2. Save scaler stats (for reference in operator)
    scaler_path = output_dir / 'memstream_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'mean': ms_core.mean.cpu(),
            'std': ms_core.std.cpu(),
        }, f)
    LOGGER.info(f"Saved scaler to {scaler_path}")

    # 3. Save memory state
    memory_path = output_dir / 'memstream_memory.pt'
    memory_state = {
        'memory': ms_core.memory.memory.cpu(),
        'memory_count': ms_core.memory.count,
        'memory_ptr': ms_core.memory.mem_ptr,
        'count': ms_core.count,
        'max_thres': ms_core.max_thres.item() if hasattr(ms_core.max_thres, 'item') else ms_core.max_thres,
        'eval_mode': ms_core.eval_mode,
    }
    torch.save(memory_state, memory_path)
    LOGGER.info(f"Saved memory state to {memory_path}")

    # 4. Save config metadata
    config_path = output_dir / 'memstream_config.json'
    config_meta = {
        'in_dim': cfg.in_dim,
        'hidden_dim': cfg.hidden_dim,
        'out_dim': cfg.out_dim,
        'memory_len': cfg.memory_len,
        'warmup_epochs': cfg.warmup_epochs,
        'warmup_batch_size': cfg.warmup_batch_size,
        'warmup_noise_std': cfg.warmup_noise_std,
        'default_beta': cfg.default_beta,
        'k_neighbors': cfg.k_neighbors,
        'seed': cfg.seed,
        'warmup_completed': True,
    }
    with open(config_path, 'w') as f:
        json.dump(config_meta, f, indent=2)
    LOGGER.info(f"Saved config to {config_path}")

    LOGGER.info(f"\nAll checkpoint files saved to {output_dir}")
    LOGGER.info("Run your Flink job to load these checkpoints.")


def main():
    """Main entry point."""
    args = parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("MemStream Offline Warmup Script")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Data file: {args.data}")
    LOGGER.info(f"Output dir: {args.output}")
    LOGGER.info(f"Epochs: {args.epochs}")
    LOGGER.info(f"Memory length: {args.memory_len}")
    LOGGER.info(f"Seed: {args.seed}")

    try:
        # Initialize vectorizer
        vectorizer = FeatureVectorizer25D()

        # Load and extract features
        features = load_data(args.data, vectorizer)

        if len(features) < 1000:
            LOGGER.warning(
                f"Only {len(features)} samples loaded. "
                "Consider using more data for better warmup."
            )

        # Train MemStream
        ms_core, cfg = train_memstream(features, args)

        # Save checkpoints
        save_checkpoint(ms_core, cfg, args)

        LOGGER.info("=" * 60)
        LOGGER.info("Warmup complete!")
        LOGGER.info("=" * 60)

    except FileNotFoundError as e:
        LOGGER.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
