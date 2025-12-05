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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from ml.evaluate import hit_ratio_at_k, ndcg_at_k, recall_at_k
from ml.model import create_hybrid_vae, vae_loss_function
from ml.train import UserInteractionDataset, get_user_indices_from_df, load_training_data
from preprocessing.embeddings import load_embeddings
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Default Search Space
# =============================================================================

DEFAULT_SEARCH_SPACE = {
    "latent_dim": [32, 64, 128],
    "hidden_dims": [[256], [512], [256, 128]],
    "dropout": [0.3, 0.5],
    "beta": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
}


# =============================================================================
# Training & Evaluation for Tuning
# =============================================================================


def train_single_config(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int = 10,
    patience: int = 3,
) -> Tuple[float, int]:
    """
    Train a model with given config and return best validation loss.

    Uses early stopping for efficiency during tuning.

    Returns:
        Tuple of (best_val_loss, best_epoch)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            x = batch_data.to(device)
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

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data.to(device)
                recon_x, mu, logvar = model(x)
                loss, _, _ = vae_loss_function(recon_x, x, mu, logvar, model.beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_loss, best_epoch


def evaluate_config_on_val(
    model,
    val_interaction_matrix: csr_matrix,
    val_df: pd.DataFrame,
    train_interaction_matrix: csr_matrix,
    user_to_idx: Dict,
    item_to_idx: Dict,
    device: torch.device,
    n_negatives: int = 99,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation set using negative sampling protocol.

    Returns dictionary with Recall@10, NDCG@10, Hit Ratio@10.
    """
    if k_values is None:
        k_values = [10]

    model.eval()
    n_items = val_interaction_matrix.shape[1]
    all_items = np.arange(n_items)

    results = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}

    with torch.no_grad():
        for _, row in val_df.iterrows():
            user_id = row["user_id"]
            test_item_id = row["asin"]

            if user_id not in user_to_idx or test_item_id not in item_to_idx:
                continue

            user_idx = user_to_idx[user_id]
            test_item_idx = item_to_idx[test_item_id]

            # Get user's training interactions
            seen_items = set(train_interaction_matrix[user_idx].indices)

            # Get model scores
            user_vector = (
                torch.FloatTensor(train_interaction_matrix[user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )
            scores, _, _ = model(user_vector)
            scores = scores.cpu().numpy().flatten()

            # Negative sampling
            candidate_mask = np.ones(n_items, dtype=bool)
            candidate_mask[list(seen_items)] = False
            candidate_mask[test_item_idx] = False

            available_negatives = all_items[candidate_mask]
            if len(available_negatives) < n_negatives:
                negative_items = available_negatives
            else:
                negative_items = np.random.choice(
                    available_negatives, size=n_negatives, replace=False
                )

            candidate_items = np.concatenate([[test_item_idx], negative_items])
            candidate_scores = scores[candidate_items]
            ranked_indices = np.argsort(candidate_scores)[::-1]
            ranked_items = candidate_items[ranked_indices]

            relevant_indices = np.array([test_item_idx])

            for k in k_values:
                k_recs = ranked_items[:k]
                results[k]["recall"].append(recall_at_k(k_recs, relevant_indices, k))
                results[k]["ndcg"].append(ndcg_at_k(k_recs, relevant_indices, k))
                results[k]["hit_ratio"].append(hit_ratio_at_k(k_recs, relevant_indices, k))

    # Aggregate
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = np.mean(results[k]["recall"]) if results[k]["recall"] else 0.0
        metrics[f"ndcg@{k}"] = np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else 0.0
        metrics[f"hit_ratio@{k}"] = (
            np.mean(results[k]["hit_ratio"]) if results[k]["hit_ratio"] else 0.0
        )

    return metrics


# =============================================================================
# Grid Search
# =============================================================================


def run_grid_search(
    data_dir: str,
    embeddings_path: str,
    output_dir: str,
    search_space: Optional[Dict[str, List]] = None,
    epochs_per_config: int = 10,
    patience: int = 3,
    batch_size: int = 512,
    use_annealing: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run grid search over hyperparameter space.

    Args:
        data_dir: Directory containing processed dataset
        embeddings_path: Path to item embeddings
        output_dir: Directory to save results
        search_space: Dictionary mapping param names to lists of values
        epochs_per_config: Max epochs per configuration
        patience: Early stopping patience
        batch_size: Batch size for training
        use_annealing: Whether to use beta annealing
        device: Device to use

    Returns:
        Dictionary with best config and all results
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    full_interaction_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]
    n_items = full_interaction_matrix.shape[1]

    # Build training matrix
    train_positives = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )
    rows = train_positives["user_id"].map(user_to_idx)
    cols = train_positives["asin"].map(item_to_idx)
    data = np.ones(len(train_positives))
    train_interaction_matrix = csr_matrix((data, (rows, cols)), shape=full_interaction_matrix.shape)

    # Build validation matrix
    val_positives = (
        val_df[val_df["binary_rating"] == 1] if "binary_rating" in val_df.columns else val_df
    )
    rows = val_positives["user_id"].map(user_to_idx)
    cols = val_positives["asin"].map(item_to_idx)
    data = np.ones(len(val_positives))
    val_interaction_matrix = csr_matrix((data, (rows, cols)), shape=full_interaction_matrix.shape)

    # Load embeddings
    embeddings_path_obj = Path(embeddings_path)
    mappings_path = embeddings_path_obj.with_name(embeddings_path_obj.stem + "_mappings.pkl")
    embeddings, _, _ = load_embeddings(embeddings_path, str(mappings_path))

    # Create data loaders
    train_user_indices = get_user_indices_from_df(train_df, user_to_idx)
    val_user_indices = get_user_indices_from_df(val_df, user_to_idx)

    train_dataset = UserInteractionDataset(train_interaction_matrix, train_user_indices)
    val_dataset = UserInteractionDataset(val_interaction_matrix, val_user_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Generate all configurations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_configs = list(itertools.product(*param_values))

    logger.info(f"Grid search over {len(all_configs)} configurations")
    logger.info(f"Search space: {search_space}")

    # Run grid search
    results = []
    best_config = None
    best_metric = -float("inf")

    for i, config_values in enumerate(tqdm(all_configs, desc="Grid Search")):
        config_dict = dict(zip(param_names, config_values))

        logger.info(f"\n[{i+1}/{len(all_configs)}] Testing config: {config_dict}")

        try:
            # Create model
            model = create_hybrid_vae(
                n_items=n_items,
                item_embeddings=embeddings,
                latent_dim=config_dict.get("latent_dim", 64),
                hidden_dims=config_dict.get("hidden_dims", [256]),
                dropout=config_dict.get("dropout", 0.5),
                beta=config_dict.get("beta", 0.2),
                use_annealing=use_annealing,
                anneal_steps=len(train_loader) * epochs_per_config // 2,
            )

            # Train
            val_loss, best_epoch = train_single_config(
                model,
                train_loader,
                val_loader,
                device,
                learning_rate=config_dict.get("learning_rate", 1e-3),
                epochs=epochs_per_config,
                patience=patience,
            )

            # Evaluate on validation with ranking metrics
            metrics = evaluate_config_on_val(
                model,
                val_interaction_matrix,
                val_df,
                train_interaction_matrix,
                user_to_idx,
                item_to_idx,
                device,
                n_negatives=99,
                k_values=[10],
            )

            result = {
                "config": config_dict,
                "val_loss": val_loss,
                "best_epoch": best_epoch,
                **metrics,
            }
            results.append(result)

            # Track best by NDCG@10
            if metrics["ndcg@10"] > best_metric:
                best_metric = metrics["ndcg@10"]
                best_config = config_dict

            logger.info(
                f"  Val Loss: {val_loss:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}, "
                f"Recall@10: {metrics['recall@10']:.4f}"
            )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"config": config_dict, "error": str(e)})

    # Sort results by NDCG@10
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["ndcg@10"], reverse=True)

    # Save results
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

    logger.info("\nGrid search complete!")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"\nBest config: {best_config}")
    logger.info(f"Best NDCG@10: {best_metric:.4f}")

    # Print top 5 configs
    logger.info("\nTop 5 configurations:")
    for i, r in enumerate(valid_results[:5]):
        logger.info(
            f"  {i+1}. NDCG@10={r['ndcg@10']:.4f}, Recall@10={r['recall@10']:.4f}, "
            f"config={r['config']}"
        )

    return {
        "best_config": best_config,
        "best_metric": best_metric,
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Hybrid VAE")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="Data directory")
    parser.add_argument("--embeddings", default=str(config.EMBEDDINGS_FILE), help="Embeddings path")
    parser.add_argument("--output", default=str(config.MODEL_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs per configuration")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Device to use")

    # Search space overrides
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Latent dimensions to search",
    )
    parser.add_argument(
        "--dropouts", type=float, nargs="+", default=[0.3, 0.5], help="Dropout rates to search"
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="Beta values to search"
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4],
        help="Learning rates to search",
    )

    args = parser.parse_args()

    # Build search space from args
    search_space = {
        "latent_dim": args.latent_dims,
        "hidden_dims": [[256], [512], [256, 128]],
        "dropout": args.dropouts,
        "beta": args.betas,
        "learning_rate": args.learning_rates,
    }

    run_grid_search(
        data_dir=args.data,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        search_space=search_space,
        epochs_per_config=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
