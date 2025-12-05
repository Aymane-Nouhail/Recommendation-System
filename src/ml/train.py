"""
Training script for Hybrid VAE recommendation system.

Handles training loop, loss tracking, and model checkpointing.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from ml.model import HybridVAE, create_hybrid_vae, vae_loss_function
from preprocessing.embeddings import load_embeddings
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================


class UserInteractionDataset(Dataset):
    """Dataset yielding user interaction vectors for VAE training."""

    def __init__(self, interaction_matrix: csr_matrix, user_indices: list[int] | None = None):
        self.interaction_matrix = interaction_matrix
        self.user_indices = user_indices or list(range(interaction_matrix.shape[0]))

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        user_vector = self.interaction_matrix[self.user_indices[idx]].toarray().flatten()
        return torch.FloatTensor(user_vector)


# =============================================================================
# Trainer
# =============================================================================


class VAETrainer:
    """Trainer class for Hybrid VAE model."""

    def __init__(
        self, model: HybridVAE, device: torch.device, lr: float = 0.001, weight_decay: float = 0.0
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_recon_losses: list[float] = []
        self.train_kl_losses: list[float] = []

        logger.info(f"Trainer on {device}, {sum(p.numel() for p in model.parameters()):,} params")

    def _compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass and loss computation."""
        recon_x, mu, logvar = self.model(x)
        if hasattr(self.model, "compute_loss"):
            loss, recon, kl = self.model.compute_loss(recon_x, x, mu, logvar)
            self.model.step_annealing()
        else:
            loss, recon, kl = vae_loss_function(recon_x, x, mu, logvar, self.model.beta)
        return loss, recon, kl

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for batch in tqdm(loader, desc="Training"):
            x = batch.to(self.device)
            self.optimizer.zero_grad()
            loss, recon, kl = self._compute_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(loader)
        return {
            "total_loss": total_loss / n,
            "recon_loss": total_recon / n,
            "kl_loss": total_kl / n,
        }

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)
                recon_x, mu, logvar = self.model(x)
                loss, recon, kl = vae_loss_function(recon_x, x, mu, logvar, self.model.beta)
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        n = len(loader)
        return {
            "total_loss": total_loss / n,
            "recon_loss": total_recon / n,
            "kl_loss": total_kl / n,
        }

    def save_checkpoint(
        self, path: str | Path, epoch: int, is_best: bool = False, extra: dict | None = None
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_recon_losses": self.train_recon_losses,
            "train_kl_losses": self.train_kl_losses,
            **(extra or {}),
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = Path(path).parent / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")


# =============================================================================
# Data Loading
# =============================================================================


def load_training_data(data_dir: str) -> tuple[csr_matrix, pd.DataFrame, pd.DataFrame, dict]:
    """Load training data from processed dataset."""
    path = Path(data_dir)

    with open(path / "interaction_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)

    train_df = pd.read_csv(path / "train.csv", low_memory=False)
    val_df = pd.read_csv(path / "val.csv", low_memory=False)

    with open(path / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    logger.info(f"Loaded: matrix {matrix.shape}, train {len(train_df)}, val {len(val_df)}")
    return matrix, train_df, val_df, mappings


def get_user_indices_from_df(df: pd.DataFrame, user_to_idx: dict[str, int]) -> list[int]:
    """Get unique user indices from DataFrame."""
    return [user_to_idx[uid] for uid in df["user_id"].unique() if uid in user_to_idx]


def _build_matrix(
    df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict, shape: tuple
) -> csr_matrix:
    """Build sparse interaction matrix from DataFrame."""
    positives = df[df["binary_rating"] == 1] if "binary_rating" in df.columns else df
    rows = positives["user_id"].map(user_to_idx)
    cols = positives["asin"].map(item_to_idx)
    return csr_matrix((np.ones(len(positives)), (rows, cols)), shape=shape)


def _get_device(device: str | None = None) -> torch.device:
    """Get best available device."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Main Training Function
# =============================================================================


def train_hybrid_vae(
    data_dir: str,
    embeddings_path: str,
    output_dir: str,
    latent_dim: int = 200,
    hidden_dims: list[int] | None = None,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    beta: float = 0.2,
    dropout: float = 0.5,
    use_annealing: bool = False,
    patience: int = 10,
    device: str | None = None,
    ignore_embeddings: bool = False,
) -> None:
    """Train Hybrid VAE model."""
    dev = _get_device(device)
    logger.info(f"Using device: {dev}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    full_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx, item_to_idx = mappings["user_to_idx"], mappings["item_to_idx"]
    n_items = full_matrix.shape[1]

    # Build train/val matrices
    train_matrix = _build_matrix(train_df, user_to_idx, item_to_idx, full_matrix.shape)
    val_matrix = _build_matrix(val_df, user_to_idx, item_to_idx, full_matrix.shape)

    # Load embeddings
    emb_path = Path(embeddings_path)
    mappings_path = emb_path.with_name(f"{emb_path.stem}_mappings.pkl")
    embeddings, emb_item_to_idx, _ = load_embeddings(embeddings_path, str(mappings_path))

    assert (
        emb_item_to_idx and len(emb_item_to_idx) == n_items
    ), f"Embedding mismatch: {len(emb_item_to_idx) if emb_item_to_idx else 0} vs {n_items}"

    if ignore_embeddings:
        logger.info("Using random embeddings instead of SBERT")
        embeddings = np.random.normal(0, 0.01, embeddings.shape).astype(np.float32)

    # Create data loaders
    train_loader = DataLoader(
        UserInteractionDataset(train_matrix, get_user_indices_from_df(train_df, user_to_idx)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        UserInteractionDataset(val_matrix, get_user_indices_from_df(val_df, user_to_idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    anneal_steps = int(len(train_loader) * epochs * 0.5)
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
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Training loop
    trainer = VAETrainer(model, dev, learning_rate, weight_decay)
    best_val_loss, patience_counter = float("inf"), 0

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        # Store history
        trainer.train_losses.append(train_metrics["total_loss"])
        trainer.val_losses.append(val_metrics["total_loss"])
        trainer.train_recon_losses.append(train_metrics["recon_loss"])
        trainer.train_kl_losses.append(train_metrics["kl_loss"])

        logger.info(
            f"Train: {train_metrics['total_loss']:.4f} (recon={train_metrics['recon_loss']:.4f}, kl={train_metrics['kl_loss']:.4f})"
        )
        logger.info(f"Val: {val_metrics['total_loss']:.4f}")

        # Checkpointing
        is_best = val_metrics["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        trainer.save_checkpoint(
            output_path / f"checkpoint_epoch_{epoch + 1}.pth",
            epoch + 1,
            is_best,
            extra={
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

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Save history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(
            {
                "train_losses": trainer.train_losses,
                "val_losses": trainer.val_losses,
                "train_recon_losses": trainer.train_recon_losses,
                "train_kl_losses": trainer.train_kl_losses,
            },
            f,
            indent=2,
        )

    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hybrid VAE")
    parser.add_argument("--data", default=str(config.DATA_DIR))
    parser.add_argument("--embeddings", default=config.EMBEDDINGS_FILE)
    parser.add_argument("--output", default=str(config.MODEL_DIR))
    parser.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[config.HIDDEN_DIM])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-annealing", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--ignore-embeddings", action="store_true")

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
