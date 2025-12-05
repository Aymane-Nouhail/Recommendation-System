"""
Evaluation module for recommendation system.

Implements Recall@K, NDCG@K, and Hit Ratio@K metrics for leave-one-out evaluation.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ml.model import HybridVAE, create_hybrid_vae
from ml.train import load_training_data
from preprocessing.embeddings import load_embeddings
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================


def recall_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Recall@K: fraction of relevant items in top-k."""
    if len(relevant) == 0:
        return 0.0
    hits = len(np.intersect1d(recommended[:k], relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    if len(relevant) == 0:
        return 0.0

    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_ratio_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Hit Ratio@K: 1 if any relevant item in top-k, else 0."""
    if len(relevant) == 0:
        return 0.0
    return 1.0 if len(np.intersect1d(recommended[:k], relevant)) > 0 else 0.0


# =============================================================================
# Helpers
# =============================================================================


def _get_device(device: str | None = None) -> torch.device:
    """Get best available device."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_input_matrix(
    train_df: pd.DataFrame, val_df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict, shape: tuple
) -> csr_matrix:
    """Build sparse interaction matrix from train/val data."""
    train_pos = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )
    val_pos = val_df[val_df["binary_rating"] == 1] if "binary_rating" in val_df.columns else val_df

    combined = pd.concat([train_pos, val_pos])
    rows = combined["user_id"].map(user_to_idx)
    cols = combined["asin"].map(item_to_idx)
    return csr_matrix((np.ones(len(combined)), (rows, cols)), shape=shape)


def _aggregate_metrics(all_metrics: dict, k_values: list[int]) -> dict[int, dict[str, float]]:
    """Aggregate collected metrics into averages."""
    return {
        k: {
            metric: float(np.mean(all_metrics[k][metric])) if all_metrics[k][metric] else 0.0
            for metric in ["recall", "ndcg", "hit_ratio"]
        }
        for k in k_values
    }


# =============================================================================
# Evaluator
# =============================================================================


class RecommendationEvaluator:
    """Evaluator for recommendation systems using leave-one-out methodology."""

    def __init__(
        self,
        model: HybridVAE,
        interaction_matrix: csr_matrix,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.interaction_matrix = interaction_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.n_items = interaction_matrix.shape[1]

    def _get_user_scores(self, user_idx: int) -> np.ndarray:
        """Get model scores for all items for a user."""
        user_vec = (
            torch.FloatTensor(self.interaction_matrix[user_idx].toarray().flatten())
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            embedding = self.model.get_user_embedding(user_vec)
            return self.model.decode(embedding).squeeze().cpu().numpy()

    def get_user_recommendations(
        self, user_idx: int, top_k: int = 100, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get top-k recommendations for a user (full ranking)."""
        scores = self._get_user_scores(user_idx)

        if exclude_seen:
            scores[self.interaction_matrix[user_idx].nonzero()[1]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, scores[top_indices]

    def evaluate_user_with_negatives(
        self,
        user_idx: int,
        test_item_idx: int,
        n_negatives: int = 99,
        k_values: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        """Evaluate single user with negative sampling protocol."""
        k_values = k_values or [5, 10, 20]

        # Sample negatives
        seen = set(self.interaction_matrix[user_idx].indices)
        mask = np.ones(self.n_items, dtype=bool)
        mask[list(seen)] = False
        mask[test_item_idx] = False

        available = np.where(mask)[0]
        negatives = (
            available
            if len(available) < n_negatives
            else np.random.choice(available, n_negatives, replace=False)
        )

        # Rank candidates
        candidates = np.concatenate([[test_item_idx], negatives])
        scores = self._get_user_scores(user_idx)
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        relevant = np.array([test_item_idx])

        return {
            k: {
                "recall": recall_at_k(ranked, relevant, k),
                "ndcg": ndcg_at_k(ranked, relevant, k),
                "hit_ratio": hit_ratio_at_k(ranked, relevant, k),
            }
            for k in k_values
        }

    def evaluate_dataset_with_negatives(
        self,
        test_df: pd.DataFrame,
        n_negatives: int = 99,
        k_values: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        """Evaluate with negative sampling protocol (standard NCF evaluation)."""
        k_values = k_values or [5, 10, 20]
        logger.info(f"Evaluating with negative sampling ({n_negatives} negatives)...")

        all_metrics = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}
        evaluated = 0

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            user_id, item_id = row["user_id"], row["asin"]
            if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                continue

            user_metrics = self.evaluate_user_with_negatives(
                self.user_to_idx[user_id], self.item_to_idx[item_id], n_negatives, k_values
            )

            evaluated += 1
            for k in k_values:
                for metric in ["recall", "ndcg", "hit_ratio"]:
                    all_metrics[k][metric].append(user_metrics[k][metric])

        logger.info(f"Evaluated {evaluated} users")
        return _aggregate_metrics(all_metrics, k_values)

    def evaluate_user(
        self, user_id: str, test_items: list[str], k_values: list[int] | None = None
    ) -> dict[int, dict[str, float]]:
        """Evaluate recommendations for a single user (full ranking)."""
        k_values = k_values or [5, 10, 20]

        if user_id not in self.user_to_idx:
            return {}

        test_indices = np.array([self.item_to_idx[i] for i in test_items if i in self.item_to_idx])
        if len(test_indices) == 0:
            return {}

        recommended, _ = self.get_user_recommendations(
            self.user_to_idx[user_id], top_k=max(k_values)
        )

        return {
            k: {
                "recall": recall_at_k(recommended, test_indices, k),
                "ndcg": ndcg_at_k(recommended, test_indices, k),
                "hit_ratio": hit_ratio_at_k(recommended, test_indices, k),
            }
            for k in k_values
        }

    def evaluate_dataset(
        self, test_df: pd.DataFrame, k_values: list[int] | None = None
    ) -> dict[int, dict[str, float]]:
        """Evaluate with full ranking protocol."""
        k_values = k_values or [5, 10, 20]
        logger.info("Evaluating with full ranking...")

        test_by_user = test_df.groupby("user_id")["asin"].apply(list).to_dict()
        all_metrics = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}
        evaluated = 0

        for user_id, test_items in tqdm(test_by_user.items()):
            user_metrics = self.evaluate_user(user_id, test_items, k_values)
            if not user_metrics:
                continue

            evaluated += 1
            for k in k_values:
                for metric in ["recall", "ndcg", "hit_ratio"]:
                    all_metrics[k][metric].append(user_metrics[k][metric])

        logger.info(f"Evaluated {evaluated} users")
        return _aggregate_metrics(all_metrics, k_values)


# =============================================================================
# Main Functions
# =============================================================================


def load_model_from_checkpoint(
    checkpoint_path: str, item_embeddings: np.ndarray, device: torch.device
) -> HybridVAE:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["model_config"]

    model = create_hybrid_vae(
        n_items=cfg["n_items"],
        item_embeddings=item_embeddings,
        latent_dim=cfg["latent_dim"],
        hidden_dims=cfg.get("hidden_dims"),
        dropout=cfg.get("dropout", 0.5),
        beta=cfg.get("beta", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def evaluate_recommendation_model(
    model_path: str,
    data_dir: str,
    embeddings_path: str,
    k_values: list[int] | None = None,
    device: str | None = None,
    n_negatives: int | None = None,
) -> dict:
    """Main evaluation function."""
    k_values = k_values or [5, 10, 20]
    device = _get_device(device)
    logger.info(f"Using device: {device}")

    # Load data
    full_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx, item_to_idx = mappings["user_to_idx"], mappings["item_to_idx"]

    input_matrix = _build_input_matrix(
        train_df, val_df, user_to_idx, item_to_idx, full_matrix.shape
    )
    logger.info(
        f"Input matrix density: {input_matrix.nnz / (input_matrix.shape[0] * input_matrix.shape[1]):.6f}"
    )

    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    embeddings, _, _ = load_embeddings(embeddings_path)
    model = load_model_from_checkpoint(model_path, embeddings, device)

    evaluator = RecommendationEvaluator(model, input_matrix, user_to_idx, item_to_idx, device)

    # Evaluate
    if n_negatives is not None:
        protocol = f"NEGATIVE SAMPLING ({n_negatives} negatives)"
        results = evaluator.evaluate_dataset_with_negatives(test_df, n_negatives, k_values)
    else:
        protocol = "FULL RANKING (all items)"
        results = evaluator.evaluate_dataset(test_df, k_values)

    # Print results
    logger.info(f"\n{'='*70}\nHYBRID VAE EVALUATION RESULTS\nProtocol: {protocol}\n{'='*70}")
    print(f"\n{'-'*70}\n{'K':<5} | {'Recall':>12} | {'NDCG':>12} | {'Hit Ratio':>12}\n{'-'*70}")
    for k in k_values:
        m = results[k]
        print(f"@{k:<4} | {m['recall']:>12.4f} | {m['ndcg']:>12.4f} | {m['hit_ratio']:>12.4f}")
    print("-" * 70)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--model", default=config.MODEL_FILE)
    parser.add_argument("--data", default=str(config.DATA_DIR))
    parser.add_argument("--embeddings", default=config.EMBEDDINGS_FILE)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--output", help="Path to save results (JSON)")
    parser.add_argument(
        "--n-negatives", type=int, default=99, help="Negatives count (0 for full ranking)"
    )

    args = parser.parse_args()
    n_negatives = args.n_negatives if args.n_negatives > 0 else None

    results = evaluate_recommendation_model(
        model_path=args.model,
        data_dir=args.data,
        embeddings_path=args.embeddings,
        k_values=args.k_values,
        device=args.device,
        n_negatives=n_negatives,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
