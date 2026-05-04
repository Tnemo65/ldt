"""
Autoencoder and VAE models for anomaly detection via reconstruction error.

This module implements two deep learning approaches for anomaly detection:
1. Standard Autoencoder: Uses reconstruction error as anomaly score
2. Variational Autoencoder (VAE): Probabilistic approach with KL divergence

Both models use a 3-layer encoder/decoder architecture and detect anomalies
by measuring reconstruction error - high error indicates anomalous patterns.

Architecture:
    Autoencoder:
        Encoder: Input → Dense(32, relu) → Dense(16, relu) → Dense(encoding_dim, relu)
        Decoder: Dense(encoding_dim) → Dense(16, relu) → Dense(32, relu) → Dense(input_dim, linear)
        Loss: MSE (reconstruction error)

    VAE:
        Encoder: Input → Dense(32, relu) → Dense(16, relu) → z_mean, z_log_var
        Sampling: z = z_mean + exp(z_log_var/2) * epsilon (epsilon ~ N(0,1))
        Decoder: z → Dense(16, relu) → Dense(32, relu) → Dense(input_dim, linear)
        Loss: Reconstruction MSE + KL divergence

References:
    - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
      Chapter 14: Autoencoders.
    - Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
      In International Conference on Learning Representations (ICLR).

Usage:
    # Train Autoencoder
    model, scaler = train_autoencoder(X_train, encoding_dim=8, epochs=50)
    scores = predict_autoencoder(model, scaler, X_test)

    # Train VAE
    model, scaler = train_vae(X_train, latent_dim=8, epochs=50)
    scores = predict_vae(model, scaler, X_test)

    # Command line batch training
    python exp_ml_autoencoder.py --model autoencoder --input data.csv --output results/
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def train_autoencoder(
    X_train: pd.DataFrame,
    encoding_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 32,
    contamination: float = 0.01
) -> Tuple[keras.Model, StandardScaler]:
    """
    Train a standard Autoencoder for anomaly detection.

    The autoencoder learns to compress data into a lower-dimensional encoding
    and reconstruct it. Anomalies are detected by high reconstruction error.

    Architecture follows Goodfellow et al. (2016) with 3-layer encoder/decoder:
        - Encoder: Input → 32 → 16 → encoding_dim
        - Decoder: encoding_dim → 16 → 32 → Input

    Args:
        X_train: Training data (normal samples only)
        encoding_dim: Dimensionality of the latent encoding (bottleneck)
        epochs: Number of training epochs
        batch_size: Batch size for training
        contamination: Expected proportion of anomalies in the dataset (default: 0.01).
            Note: Currently reserved for future use in threshold calculation during
            anomaly detection. Not used during training phase.

    Returns:
        Tuple[keras.Model, StandardScaler]: Trained Keras model and fitted StandardScaler

    References:
        Goodfellow et al. (2016). Deep Learning. MIT Press. Chapter 14.
    """
    # NOTE: contamination parameter is reserved for future use in threshold calculation
    # during anomaly detection (e.g., determining cutoff for binary classification).
    # It is not used during the training phase of autoencoders.
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    input_dim = X_scaled.shape[1]

    # Build Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(encoder_input)
    encoded = layers.Dense(16, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Build Decoder
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    # Create model
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0
    )

    return autoencoder, scaler


def predict_autoencoder(
    model: keras.Model,
    scaler: StandardScaler,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Compute anomaly scores using trained Autoencoder.

    Anomaly score is the Mean Squared Error (MSE) between input and reconstruction.
    Higher scores indicate more anomalous samples.

    Args:
        model: Trained autoencoder model
        scaler: Fitted StandardScaler from training
        X_test: Test data to score

    Returns:
        Array of anomaly scores (reconstruction errors)
    """
    X_scaled = scaler.transform(X_test)
    reconstructions = model.predict(X_scaled, verbose=0)

    # Compute MSE per sample
    mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

    return mse


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """
    Sampling layer for VAE using the reparameterization trick.

    Implements z = mean + exp(log_var/2) * epsilon, where epsilon ~ N(0,1).
    This allows gradients to flow through the sampling operation.

    References:
        Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
    """

    def call(self, inputs):
        """Sample from latent distribution using reparameterization trick."""
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def train_vae(
    X_train: pd.DataFrame,
    latent_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 32,
    contamination: float = 0.01
) -> Tuple[keras.Model, StandardScaler]:
    """
    Train a Variational Autoencoder (VAE) for anomaly detection.

    VAE is a probabilistic generative model that learns a latent distribution.
    It uses the reparameterization trick for backpropagation and combines
    reconstruction loss with KL divergence regularization.

    Architecture follows Kingma & Welling (2014):
        - Encoder: Input → 32 → 16 → (z_mean, z_log_var)
        - Sampling: z = z_mean + exp(z_log_var/2) * epsilon
        - Decoder: z → 16 → 32 → Input
        - Loss: MSE reconstruction + KL divergence

    Args:
        X_train: Training data (normal samples only)
        latent_dim: Dimensionality of the latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
        contamination: Expected proportion of anomalies in the dataset (default: 0.01).
            Note: Currently reserved for future use in threshold calculation during
            anomaly detection. Not used during training phase.

    Returns:
        Tuple[keras.Model, StandardScaler]: Trained Keras model and fitted StandardScaler

    References:
        Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
        In International Conference on Learning Representations (ICLR).
    """
    # NOTE: contamination parameter is reserved for future use in threshold calculation
    # during anomaly detection (e.g., determining cutoff for binary classification).
    # It is not used during the training phase of VAEs.
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    input_dim = X_scaled.shape[1]

    # Build Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='relu')(encoder_input)
    x = layers.Dense(16, activation='relu')(x)

    # Latent space parameters
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # Sampling layer
    z = Sampling()([z_mean, z_log_var])

    # Build Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation='relu')(decoder_input)
    x = layers.Dense(32, activation='relu')(x)
    decoder_output = layers.Dense(input_dim, activation='linear')(x)

    decoder = keras.Model(decoder_input, decoder_output, name='decoder')

    # Build VAE
    vae_output = decoder(z)
    vae = keras.Model(encoder_input, vae_output, name='vae')

    # VAE loss: reconstruction + KL divergence
    # Use mean over all dimensions for numerical stability (standard VAE practice)
    reconstruction_loss = tf.reduce_mean(tf.square(encoder_input - vae_output))

    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
    )

    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam')

    # Train
    vae.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0
    )

    return vae, scaler


def predict_vae(
    model: keras.Model,
    scaler: StandardScaler,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Compute anomaly scores using trained VAE.

    Anomaly score is the reconstruction error (MSE between input and output).
    Higher scores indicate more anomalous samples.

    Args:
        model: Trained VAE model
        scaler: Fitted StandardScaler from training
        X_test: Test data to score

    Returns:
        Array of anomaly scores (reconstruction errors)
    """
    X_scaled = scaler.transform(X_test)
    reconstructions = model.predict(X_scaled, verbose=0)

    # Compute MSE per sample
    mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

    return mse


def load_vae_model(model_path: str) -> keras.Model:
    """
    Load a saved VAE model with custom Sampling layer.

    This function ensures the custom Sampling layer is properly registered
    when loading a saved VAE model.

    Args:
        model_path: Path to saved .keras model file

    Returns:
        Loaded VAE model

    Note:
        The Sampling layer is already registered via decorator, so this
        function simply wraps keras.models.load_model for convenience.
    """
    return keras.models.load_model(model_path)


def main():
    """
    Command-line interface for batch training of Autoencoder and VAE models.

    Example usage:
        python exp_ml_autoencoder.py --model autoencoder --input data.csv --output results/
        python exp_ml_autoencoder.py --model vae --latent-dim 16 --epochs 100
    """
    parser = argparse.ArgumentParser(
        description='Train Autoencoder or VAE for anomaly detection'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['autoencoder', 'vae'],
        required=True,
        help='Model type to train'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (training data)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for model and scaler'
    )

    parser.add_argument(
        '--encoding-dim',
        type=int,
        default=8,
        help='Encoding/latent dimension (default: 8)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    X_train = pd.read_csv(args.input)
    print(f"Loaded {len(X_train)} samples with {len(X_train.columns)} features")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    if args.model == 'autoencoder':
        print(f"\nTraining Autoencoder (encoding_dim={args.encoding_dim})...")
        model, scaler = train_autoencoder(
            X_train,
            encoding_dim=args.encoding_dim,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        model_path = output_dir / 'autoencoder.keras'
        scaler_path = output_dir / 'autoencoder_scaler.pkl'
    else:  # vae
        print(f"\nTraining VAE (latent_dim={args.encoding_dim})...")
        model, scaler = train_vae(
            X_train,
            latent_dim=args.encoding_dim,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        model_path = output_dir / 'vae.keras'
        scaler_path = output_dir / 'vae_scaler.pkl'

    # Save model and scaler
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)

    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
