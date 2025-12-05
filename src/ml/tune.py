"""
Hyperparameter tuning for Hybrid VAE using grid search.

Searches over key hyperparameters and evaluates on validation set.
Results are saved for analysis and best config is reported.
"""

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from ml.evaluate import hit_ratio_at_k, ndcg_at_k, recall_at_k
from ml.model import create_hybrid_vae, vae_loss_function
from ml.train import UserInteractionDataset, get_user_indices_from_df, load_training_data
from preprocessing.embeddings import load_embeddings
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE = {
    "latent_dim": [32, 64, 128],
    "hidden_dims": [[256], [512], [256, 128]],
    "dropout": [0.3, 0.5],
    "beta": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
}


def _get_device(device: str | None = None) -> torch.device:
    """Get best available device."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_interaction_matrix(
    df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict, shape: tuple
) -> csr_matrix:
    """Build sparse interaction matrix from dataframe."""
    positives = df[df["binary_rating"] == 1] if "binary_rating" in df.columns else df
    rows = positives["user_id"].map(user_to_idx)
    cols = positives["asin"].map(item_to_idx)
    return csr_matrix((np.ones(len(positives)), (rows, cols)), shape=shape)


def train_single_config(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int = 10,
    patience: int = 3,
) -> tuple[float, int]:
    """Train model and return best validation loss with early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            if hasattr(model, "compute_loss"):
                loss, _, _ = model.compute_loss(recon_x, x, mu, logvar)
                model.step_annealing()
            else:
                loss, _, _ = vae_loss_function(recon_x, x, mu, logvar, model.beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon_x, mu, logvar = model(x)
                loss, _, _ = vae_loss_function(recon_x, x, mu, logvar, model.beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch + 1, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss, best_epoch


def evaluate_config_on_val(
    model,
    train_matrix: csr_matrix,
    val_df: pd.DataFrame,
    user_to_idx: dict,
    item_to_idx: dict,
    device: torch.device,
    n_negatives: int = 99,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate model on validation set using negative sampling protocol."""
    k_values = k_values or [10]
    model.eval()
    n_items = train_matrix.shape[1]
    all_items = np.arange(n_items)

    results = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}

    with torch.no_grad():
        for _, row in val_df.iterrows():
            user_id, test_item_id = row["user_id"], row["asin"]
            if user_id not in user_to_idx or test_item_id not in item_to_idx:
                continue

            user_idx = user_to_idx[user_id]
            test_item_idx = item_to_idx[test_item_id]
            seen_items = set(train_matrix[user_idx].indices)

            # Get scores
            user_vec = (
                torch.FloatTensor(train_matrix[user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )
            scores = model(user_vec)[0].cpu().numpy().flatten()

            # Sample negatives
            candidate_mask = np.ones(n_items, dtype=bool)
            candidate_mask[list(seen_items)] = False
            candidate_mask[test_item_idx] = False
            available = all_items[candidate_mask]
            negatives = (
                available
                if len(available) < n_negatives
                else np.random.choice(available, n_negatives, replace=False)
            )

            # Rank candidates
            candidates = np.concatenate([[test_item_idx], negatives])
            ranked = candidates[np.argsort(scores[candidates])[::-1]]
            relevant = np.array([test_item_idx])

            for k in k_values:
                top_k = ranked[:k]
                results[k]["recall"].append(recall_at_k(top_k, relevant, k))
                results[k]["ndcg"].append(ndcg_at_k(top_k, relevant, k))
                results[k]["hit_ratio"].append(hit_ratio_at_k(top_k, relevant, k))

    # Aggregate metrics
    return {
        f"{metric}@{k}": float(np.mean(results[k][metric])) if results[k][metric] else 0.0
        for k in k_values
        for metric in ["recall", "ndcg", "hit_ratio"]
    }


def run_grid_search(
    data_dir: str,
    embeddings_path: str,
    output_dir: str,
    search_space: dict[str, list] | None = None,
    epochs_per_config: int = 10,
    patience: int = 3,
    batch_size: int = 512,
    use_annealing: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Run grid search over hyperparameter space."""
    search_space = search_space or DEFAULT_SEARCH_SPACE
    device = _get_device(device)
    logger.info(f"Using device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    full_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx, item_to_idx = mappings["user_to_idx"], mappings["item_to_idx"]
    n_items = full_matrix.shape[1]

    train_matrix = _build_interaction_matrix(train_df, user_to_idx, item_to_idx, full_matrix.shape)
    val_matrix = _build_interaction_matrix(val_df, user_to_idx, item_to_idx, full_matrix.shape)

    # Load embeddings
    emb_path = Path(embeddings_path)
    embeddings, _, _ = load_embeddings(
        embeddings_path, str(emb_path.with_name(f"{emb_path.stem}_mappings.pkl"))
    )

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

    # Generate configurations
    param_names = list(search_space.keys())
    all_configs = list(itertools.product(*search_space.values()))
    logger.info(f"Grid search over {len(all_configs)} configurations")

    results, best_config, best_metric = [], None, -float("inf")

    for i, values in enumerate(tqdm(all_configs, desc="Grid Search")):
        cfg = dict(zip(param_names, values))
        logger.info(f"\n[{i+1}/{len(all_configs)}] Testing: {cfg}")

        try:
            model = create_hybrid_vae(
                n_items=n_items,
                item_embeddings=embeddings,
                latent_dim=cfg.get("latent_dim", 64),
                hidden_dims=cfg.get("hidden_dims", [256]),
                dropout=cfg.get("dropout", 0.5),
                beta=cfg.get("beta", 0.2),
                use_annealing=use_annealing,
                anneal_steps=len(train_loader) * epochs_per_config // 2,
            )

            val_loss, best_epoch = train_single_config(
                model,
                train_loader,
                val_loader,
                device,
                learning_rate=cfg.get("learning_rate", 1e-3),
                epochs=epochs_per_config,
                patience=patience,
            )

            metrics = evaluate_config_on_val(
                model,
                train_matrix,
                val_df,
                user_to_idx,
                item_to_idx,
                device,
                n_negatives=99,
                k_values=[10],
            )

            result = {"config": cfg, "val_loss": val_loss, "best_epoch": best_epoch, **metrics}
            results.append(result)

            if metrics["ndcg@10"] > best_metric:
                best_metric, best_config = metrics["ndcg@10"], cfg

            logger.info(
                f"  Val Loss: {val_loss:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}, Recall@10: {metrics['recall@10']:.4f}"
            )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"config": cfg, "error": str(e)})

    # Save results
    valid_results = sorted(
        [r for r in results if "error" not in r], key=lambda x: x["ndcg@10"], reverse=True
    )

    output_file = output_path / "grid_search_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "search_space": {k: [str(v) for v in vals] for k, vals in search_space.items()},
                "best_config": best_config,
                "best_ndcg@10": best_metric,
                "all_results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"\nGrid search complete! Results saved to {output_file}")
    logger.info(f"Best config: {best_config}")
    logger.info(f"Best NDCG@10: {best_metric:.4f}")

    logger.info("\nTop 5 configurations:")
    for i, r in enumerate(valid_results[:5]):
        logger.info(
            f"  {i+1}. NDCG@10={r['ndcg@10']:.4f}, Recall@10={r['recall@10']:.4f}, config={r['config']}"
        )

    return {"best_config": best_config, "best_metric": best_metric, "all_results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Hybrid VAE")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="Data directory")
    parser.add_argument("--embeddings", default=str(config.EMBEDDINGS_FILE), help="Embeddings path")
    parser.add_argument("--output", default=str(config.MODEL_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs per config")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Device")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 5e-4])

    args = parser.parse_args()

    run_grid_search(
        data_dir=args.data,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        search_space={
            "latent_dim": args.latent_dims,
            "hidden_dims": [[256], [512], [256, 128]],
            "dropout": args.dropouts,
            "beta": args.betas,
            "learning_rate": args.learning_rates,
        },
        epochs_per_config=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
