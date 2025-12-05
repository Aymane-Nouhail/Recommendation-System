"""
Evaluation module for recommendation system.

This module implements evaluation metrics including Recall@K and NDCG@K
for leave-one-out test methodology.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.model import HybridVAE, create_hybrid_vae
from preprocessing.embeddings import load_embeddings
from ml.train import load_training_data
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recall_at_k(
    recommended_items: np.ndarray, relevant_items: np.ndarray, k: int
) -> float:
    """
    Calculate Recall@K for a single user.

    Args:
        recommended_items: Array of recommended item indices (sorted by score)
        relevant_items: Array of relevant item indices
        k: Number of top recommendations to consider

    Returns:
        Recall@K value
    """
    if len(relevant_items) == 0:
        return 0.0

    # Get top-k recommendations
    top_k_items = recommended_items[:k]

    # Count how many relevant items are in top-k
    hits = len(np.intersect1d(top_k_items, relevant_items))

    # Recall = hits / total_relevant
    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items: np.ndarray, relevant_items: np.ndarray, k: int
) -> float:
    """
    Calculate NDCG@K for a single user.

    Args:
        recommended_items: Array of recommended item indices (sorted by score)
        relevant_items: Array of relevant item indices
        k: Number of top recommendations to consider

    Returns:
        NDCG@K value
    """
    if len(relevant_items) == 0:
        return 0.0

    # Get top-k recommendations
    top_k_items = recommended_items[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item in enumerate(top_k_items):
        if item in relevant_items:
            # Binary relevance (1 if relevant, 0 otherwise)
            # DCG formula: sum(rel_i / log2(i + 2)) where i is 0-indexed
            dcg += 1.0 / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG)
    # For binary relevance, IDCG is the DCG of a perfect ranking
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)

    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0


def hit_ratio_at_k(
    recommended_items: np.ndarray, relevant_items: np.ndarray, k: int
) -> float:
    """
    Calculate Hit Ratio@K for a single user.

    Args:
        recommended_items: Array of recommended item indices
        relevant_items: Array of relevant item indices
        k: Number of top recommendations to consider

    Returns:
        Hit Ratio@K (1 if any relevant item is in top-k, 0 otherwise)
    """
    if len(relevant_items) == 0:
        return 0.0

    top_k_items = recommended_items[:k]
    return 1.0 if len(np.intersect1d(top_k_items, relevant_items)) > 0 else 0.0


class RecommendationEvaluator:
    """
    Evaluator for recommendation systems using leave-one-out methodology.
    """

    def __init__(
        self,
        model: HybridVAE,
        interaction_matrix: csr_matrix,
        user_to_idx: Dict[str, int],
        item_to_idx: Dict[str, int],
        device: torch.device,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Trained HybridVAE model
            interaction_matrix: User-item interaction matrix
            user_to_idx: User ID to index mapping
            item_to_idx: Item ID to index mapping
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.interaction_matrix = interaction_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx

        # Create reverse mappings
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        self.model.eval()

    def get_user_recommendations(
        self, user_idx: int, top_k: int = 100, exclude_seen: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recommendations for a single user.

        Args:
            user_idx: User index
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already seen

        Returns:
            Tuple of (recommended_item_indices, scores)
        """
        with torch.no_grad():
            # Get user interaction vector
            user_vector = (
                torch.FloatTensor(self.interaction_matrix[user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(self.device)
            )

            # Get user embedding
            user_embedding = self.model.get_user_embedding(user_vector)

            # Get item scores
            scores = self.model.decode(user_embedding)
            scores = scores.squeeze().cpu().numpy()

            # Exclude seen items if requested
            if exclude_seen:
                seen_items = self.interaction_matrix[user_idx].nonzero()[1]
                scores[seen_items] = -np.inf

            # Get top-k items
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[top_indices]

            return top_indices, top_scores

    def evaluate_user(
        self, user_id: str, test_items: List[str], k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate recommendations for a single user.

        Args:
            user_id: User ID
            test_items: List of test item IDs (ground truth)
            k_values: List of k values for evaluation

        Returns:
            Dictionary of metrics for different k values
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return {}

        user_idx = self.user_to_idx[user_id]

        # Convert test item IDs to indices
        test_item_indices = []
        for item_id in test_items:
            if item_id in self.item_to_idx:
                test_item_indices.append(self.item_to_idx[item_id])

        if not test_item_indices:
            return {}

        test_item_indices = np.array(test_item_indices)

        # Get recommendations (use max k for efficiency)
        max_k = max(k_values) if k_values else 100
        recommended_indices, _ = self.get_user_recommendations(user_idx, top_k=max_k)

        # Calculate metrics for each k
        metrics = {}
        for k in k_values:
            metrics[k] = {
                "recall": recall_at_k(recommended_indices, test_item_indices, k),
                "ndcg": ndcg_at_k(recommended_indices, test_item_indices, k),
                "hit_ratio": hit_ratio_at_k(recommended_indices, test_item_indices, k),
            }

        return metrics

    def evaluate_dataset(
        self, test_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the model on the test dataset.

        Args:
            test_df: Test DataFrame with user-item interactions
            k_values: List of k values for evaluation

        Returns:
            Dictionary of averaged metrics
        """
        logger.info("Evaluating model on test dataset...")

        # Group test data by user
        test_by_user = test_df.groupby("user_id")["asin"].apply(list).to_dict()

        # Collect metrics for all users
        all_metrics = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}

        evaluated_users = 0

        for user_id, test_items in tqdm(test_by_user.items()):
            user_metrics = self.evaluate_user(user_id, test_items, k_values)

            if user_metrics:
                evaluated_users += 1
                for k in k_values:
                    if k in user_metrics:
                        all_metrics[k]["recall"].append(user_metrics[k]["recall"])
                        all_metrics[k]["ndcg"].append(user_metrics[k]["ndcg"])
                        all_metrics[k]["hit_ratio"].append(user_metrics[k]["hit_ratio"])

        # Calculate averages
        avg_metrics = {}
        for k in k_values:
            avg_metrics[k] = {
                "recall": (
                    np.mean(all_metrics[k]["recall"])
                    if all_metrics[k]["recall"]
                    else 0.0
                ),
                "ndcg": (
                    np.mean(all_metrics[k]["ndcg"]) if all_metrics[k]["ndcg"] else 0.0
                ),
                "hit_ratio": (
                    np.mean(all_metrics[k]["hit_ratio"])
                    if all_metrics[k]["hit_ratio"]
                    else 0.0
                ),
            }

        logger.info(f"Evaluated {evaluated_users} users")

        return avg_metrics


def load_model_from_checkpoint(
    checkpoint_path: str, item_embeddings: np.ndarray, device: torch.device
) -> HybridVAE:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        item_embeddings: Item embeddings matrix
        device: Device to load model on

    Returns:
        Loaded HybridVAE model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Create model
    model = create_hybrid_vae(
        n_items=model_config["n_items"],
        item_embeddings=item_embeddings,
        latent_dim=model_config["latent_dim"],
        hidden_dims=model_config.get("hidden_dims"),
        dropout=model_config.get("dropout", 0.5),
        beta=model_config.get("beta", 0.2),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")

    return model


def evaluate_recommendation_model(
    model_path: str,
    data_dir: str,
    embeddings_path: str,
    k_values: List[int] = [5, 10, 20],
    device: Optional[str] = None,
) -> Dict:
    """
    Main evaluation function.

    Args:
        model_path: Path to trained model
        data_dir: Directory containing processed dataset
        embeddings_path: Path to item embeddings
        k_values: List of k values for evaluation
        device: Device to use for evaluation

    Returns:
        Dictionary with evaluation results
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load data
    full_interaction_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    # Reconstruct input matrix from train and val data only (exclude test data)
    # This ensures we don't mask the test items we are trying to predict
    logger.info("Reconstructing input matrix from train/val data...")

    # Filter positives only (if negatives were added)
    train_positives = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )
    val_positives = (
        val_df[val_df["binary_rating"] == 1]
        if "binary_rating" in val_df.columns
        else val_df
    )

    input_df = pd.concat([train_positives, val_positives])

    # Build sparse matrix
    rows = input_df["user_id"].map(user_to_idx)
    cols = input_df["asin"].map(item_to_idx)
    data = np.ones(len(input_df))

    # Use shape from full matrix to ensure dimensions match
    input_matrix = csr_matrix((data, (rows, cols)), shape=full_interaction_matrix.shape)

    logger.info(
        f"Input matrix density: {input_matrix.nnz / (input_matrix.shape[0] * input_matrix.shape[1]):.6f}"
    )

    # Load test data
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    # Load embeddings
    embeddings, _, _ = load_embeddings(embeddings_path)

    # Load model
    model = load_model_from_checkpoint(model_path, embeddings, device)

    # Create evaluator
    evaluator = RecommendationEvaluator(
        model=model,
        interaction_matrix=input_matrix,  # Use reconstructed matrix
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        device=device,
    )

    # Evaluate
    results = evaluator.evaluate_dataset(test_df, k_values)

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    for k in k_values:
        if k in results:
            metrics = results[k]
            logger.info(
                f"@{k:2d}: Recall={metrics['recall']:.4f}, "
                f"NDCG={metrics['ndcg']:.4f}, "
                f"Hit Ratio={metrics['hit_ratio']:.4f}"
            )

    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument(
        "--model", default=config.MODEL_FILE, help="Path to trained model"
    )
    parser.add_argument(
        "--data", default=str(config.DATA_DIR), help="Directory containing dataset"
    )
    parser.add_argument(
        "--embeddings", default=config.EMBEDDINGS_FILE, help="Path to item embeddings"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--output", help="Path to save evaluation results (JSON)")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_recommendation_model(
        model_path=args.model,
        data_dir=args.data,
        embeddings_path=args.embeddings,
        k_values=args.k_values,
        device=args.device,
    )

    # Save results if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
