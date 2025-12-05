"""
Training script for Hybrid VAE recommendation system.

This module handles the training loop with mini-batch users, loss tracking,
and model checkpointing.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from scipy.sparse import csr_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.model import HybridVAE, vae_loss_function, create_hybrid_vae
from preprocessing.embeddings import load_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserInteractionDataset(Dataset):
    """
    Dataset for user interactions in VAE training.

    Each sample is a user's interaction vector (binary vector over all items).
    """

    def __init__(
        self, interaction_matrix: csr_matrix, user_indices: Optional[List[int]] = None
    ):
        """
        Initialize the dataset.

        Args:
            interaction_matrix: Sparse user-item interaction matrix (users x items)
            user_indices: List of user indices to include (if None, use all users)
        """
        self.interaction_matrix = interaction_matrix
        self.user_indices = (
            user_indices
            if user_indices is not None
            else list(range(interaction_matrix.shape[0]))
        )

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get user interaction vector.

        Args:
            idx: Index in the dataset

        Returns:
            Dense interaction vector for the user
        """
        user_idx = self.user_indices[idx]

        # Convert sparse row to dense vector
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()

        return torch.FloatTensor(user_vector)


class VAETrainer:
    """
    Trainer class for Hybrid VAE model.
    """

    def __init__(
        self,
        model: HybridVAE,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """
        Initialize the trainer.

        Args:
            model: HybridVAE model to train
            device: Device to run training on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []

        logger.info(f"Initialized VAE trainer on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        n_batches = len(train_loader)

        # Progress bar
        pbar = tqdm(train_loader, desc="Training")

        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            x = batch_data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x)

            # Compute loss
            if hasattr(self.model, "compute_loss"):
                # For annealed VAE
                loss, recon_loss, kl_loss = self.model.compute_loss(
                    recon_x, x, mu, logvar
                )
                self.model.step_annealing()
            else:
                # For regular VAE
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_x, x, mu, logvar, self.model.beta
                )

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Update parameters
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        # Calculate average losses
        avg_loss = total_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches

        return {
            "total_loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        n_batches = len(val_loader)

        with torch.no_grad():
            for batch_data in val_loader:
                # Move data to device
                x = batch_data.to(self.device)

                # Forward pass
                recon_x, mu, logvar = self.model(x)

                # Compute loss
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_x, x, mu, logvar, self.model.beta
                )

                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

        # Calculate average losses
        avg_loss = total_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches

        return {
            "total_loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        is_best: bool = False,
        additional_info: Optional[Dict] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_recon_losses": self.train_recon_losses,
            "train_kl_losses": self.train_kl_losses,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = Path(filepath).parent / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")


def load_training_data(
    data_dir: str,
) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load training data from processed dataset.

    Args:
        data_dir: Directory containing processed dataset files

    Returns:
        Tuple of (interaction_matrix, train_df, val_df, mappings)
    """
    data_path = Path(data_dir)

    # Load interaction matrix
    with open(data_path / "interaction_matrix.pkl", "rb") as f:
        interaction_matrix = pickle.load(f)

    # Load train and validation DataFrames
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")

    # Load mappings
    with open(data_path / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    logger.info(f"Loaded interaction matrix: {interaction_matrix.shape}")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    return interaction_matrix, train_df, val_df, mappings


def get_user_indices_from_df(
    df: pd.DataFrame, user_to_idx: Dict[str, int]
) -> List[int]:
    """
    Get user indices from DataFrame.

    Args:
        df: DataFrame with user interactions
        user_to_idx: Mapping from user ID to index

    Returns:
        List of unique user indices
    """
    user_ids = df["user_id"].unique()
    user_indices = [
        user_to_idx[user_id] for user_id in user_ids if user_id in user_to_idx
    ]
    return user_indices


def train_hybrid_vae(
    data_dir: str,
    embeddings_path: str,
    output_dir: str,
    latent_dim: int = 200,
    hidden_dims: Optional[List[int]] = None,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    beta: float = 0.2,
    dropout: float = 0.5,
    use_annealing: bool = False,
    patience: int = 10,
    device: Optional[str] = None,
    ignore_embeddings: bool = False,
) -> None:
    """
    Main training function for Hybrid VAE.

    Args:
        data_dir: Directory containing processed dataset
        embeddings_path: Path to item embeddings
        output_dir: Directory to save trained model
        latent_dim: Latent space dimensionality
        hidden_dims: Hidden layer dimensions
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        beta: KL divergence weight
        dropout: Dropout rate
        use_annealing: Whether to use beta annealing
        patience: Early stopping patience
        device: Device to use for training
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load training data
    full_interaction_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]
    n_users, n_items = full_interaction_matrix.shape

    # Reconstruct training matrix from train_df only
    # This ensures we don't train on validation/test data
    logger.info("Reconstructing training matrix from train_df...")
    train_positives = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )

    rows = train_positives["user_id"].map(user_to_idx)
    cols = train_positives["asin"].map(item_to_idx)
    data = np.ones(len(train_positives))

    train_interaction_matrix = csr_matrix(
        (data, (rows, cols)), shape=full_interaction_matrix.shape
    )

    # Reconstruct validation matrix
    logger.info("Reconstructing validation matrix from val_df...")
    val_positives = (
        val_df[val_df["binary_rating"] == 1]
        if "binary_rating" in val_df.columns
        else val_df
    )
    rows = val_positives["user_id"].map(user_to_idx)
    cols = val_positives["asin"].map(item_to_idx)
    data = np.ones(len(val_positives))
    val_interaction_matrix = csr_matrix(
        (data, (rows, cols)), shape=full_interaction_matrix.shape
    )

    # Load item embeddings
    embeddings_path_obj = Path(embeddings_path)
    mappings_path = embeddings_path_obj.with_name(
        embeddings_path_obj.stem + "_mappings.pkl"
    )
    embeddings, item_to_idx, idx_to_item = load_embeddings(
        embeddings_path, str(mappings_path)
    )

    # Validate embeddings match dataset
    assert (
        item_to_idx is not None and len(item_to_idx) == n_items
    ), f"Embeddings ({len(item_to_idx) if item_to_idx is not None else 'None'}) don't match dataset ({n_items})"

    if ignore_embeddings:
        logger.info("Ignoring SBERT embeddings, using random initialization")
        embeddings = np.random.normal(0, 0.01, embeddings.shape).astype(np.float32)

    # Create datasets and data loaders
    train_user_indices = get_user_indices_from_df(train_df, user_to_idx)
    val_user_indices = get_user_indices_from_df(val_df, user_to_idx)

    train_dataset = UserInteractionDataset(train_interaction_matrix, train_user_indices)
    val_dataset = UserInteractionDataset(val_interaction_matrix, val_user_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Calculate annealing steps (anneal over 50% of epochs)
    total_steps = len(train_loader) * epochs
    anneal_steps = int(total_steps * 0.5)
    logger.info(f"Annealing over {anneal_steps} steps (total steps: {total_steps})")

    # Create model
    model = create_hybrid_vae(
        n_items=n_items,
        item_embeddings=embeddings,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        beta=beta,
        use_annealing=use_annealing,
        anneal_steps=anneal_steps,
    )

    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Initialize trainer
    trainer = VAETrainer(model, device, learning_rate, weight_decay)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Store losses
        trainer.train_losses.append(train_metrics["total_loss"])
        trainer.val_losses.append(val_metrics["total_loss"])
        trainer.train_recon_losses.append(train_metrics["recon_loss"])
        trainer.train_kl_losses.append(train_metrics["kl_loss"])

        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f})"
        )
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")

        # Save checkpoint
        is_best = val_metrics["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        # Save model
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pth"
        trainer.save_checkpoint(
            checkpoint_path,
            epoch + 1,
            is_best=is_best,
            additional_info={
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_config": {
                    "n_items": n_items,
                    "latent_dim": latent_dim,
                    "hidden_dims": hidden_dims,
                    "beta": beta,
                    "dropout": dropout,
                },
            },
        )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    history = {
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "train_recon_losses": trainer.train_recon_losses,
        "train_kl_losses": trainer.train_kl_losses,
    }

    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train Hybrid VAE for recommendation")
    parser.add_argument(
        "--data",
        default=str(config.DATA_DIR),
        help="Directory containing processed dataset",
    )
    parser.add_argument(
        "--embeddings", default=config.EMBEDDINGS_FILE, help="Path to item embeddings"
    )
    parser.add_argument(
        "--output",
        default=str(config.MODEL_DIR),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=config.LATENT_DIM,
        help="Latent space dimension",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[config.HIDDEN_DIM],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--beta", type=float, default=0.2, help="KL divergence weight")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--use-annealing", action="store_true", help="Use beta annealing"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument(
        "--ignore-embeddings",
        action="store_true",
        help="Ignore SBERT embeddings and use random initialization",
    )

    args = parser.parse_args()

    train_hybrid_vae(
        data_dir=args.data,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        dropout=args.dropout,
        use_annealing=args.use_annealing,
        patience=args.patience,
        device=args.device,
        ignore_embeddings=args.ignore_embeddings,
    )


if __name__ == "__main__":
    main()
