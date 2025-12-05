"""
Baseline model (Matrix Factorization) for recommendation system.

This module implements a simple Matrix Factorization baseline using TruncatedSVD
to compare with the Hybrid VAE model.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.train import load_training_data
from ml.evaluate import recall_at_k, ndcg_at_k, hit_ratio_at_k
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVDRecommender:
    """
    Matrix Factorization recommender using TruncatedSVD.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42):
        """
        Initialize the recommender.

        Args:
            n_components: Number of latent factors
            random_state: Random seed
        """
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = None
        self.item_factors = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Fit the model to the interaction matrix.

        Args:
            interaction_matrix: Sparse user-item interaction matrix
        """
        logger.info(f"Fitting SVD with {self.n_components} components...")
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_
        logger.info(
            f"Explained variance ratio: {self.model.explained_variance_ratio_.sum():.4f}"
        )

    def predict(self, user_idx: int) -> np.ndarray:
        """
        Predict scores for all items for a given user.

        Args:
            user_idx: User index

        Returns:
            Array of item scores
        """
        # Reconstruct row: user_factors[u] @ item_factors
        return self.user_factors[user_idx] @ self.item_factors


def evaluate_baseline(
    data_dir: str, n_components: int = 50, k_values: List[int] = [5, 10, 20]
):
    """
    Train and evaluate baseline model.

    Args:
        data_dir: Directory containing processed dataset
        n_components: Number of latent factors
        k_values: List of K values for evaluation
    """
    # Load data
    full_interaction_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    # Reconstruct input matrix from train and val data only (exclude test data)
    logger.info("Reconstructing input matrix from train/val data...")

    # Filter positives only
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

    # Train model
    model = SVDRecommender(n_components=n_components)
    model.fit(input_matrix)

    # Load test data
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    # Evaluate
    logger.info("Evaluating on test set...")
    results = {}
    for k in k_values:
        results[k] = {"recall": [], "ndcg": [], "hit_ratio": []}

    # Group test data by user
    test_users = test_df["user_id"].unique()

    for user_id in tqdm(test_users, desc="Evaluating"):
        if user_id not in user_to_idx:
            continue

        user_idx = user_to_idx[user_id]

        # Get relevant items (ground truth)
        user_test_items = test_df[test_df["user_id"] == user_id]["asin"].values
        relevant_indices = [
            item_to_idx[item] for item in user_test_items if item in item_to_idx
        ]

        if not relevant_indices:
            continue

        # Get predictions
        scores = model.predict(user_idx)

        # Mask training items (set score to -inf)
        # We want to recommend NEW items, not ones already seen in train/val
        user_train_items = input_df[input_df["user_id"] == user_id]["asin"].values
        train_indices = [
            item_to_idx[item] for item in user_train_items if item in item_to_idx
        ]
        scores[train_indices] = -np.inf

        # Get top K items
        # We need max K
        max_k = max(k_values)
        top_k_indices = np.argsort(scores)[::-1][:max_k]

        # Calculate metrics
        for k in k_values:
            k_recs = top_k_indices[:k]

            recall = recall_at_k(k_recs, relevant_indices, k)
            ndcg = ndcg_at_k(k_recs, relevant_indices, k)
            hit_ratio = hit_ratio_at_k(k_recs, relevant_indices, k)

            results[k]["recall"].append(recall)
            results[k]["ndcg"].append(ndcg)
            results[k]["hit_ratio"].append(hit_ratio)

    # Aggregate results
    final_metrics = {}
    logger.info("\nBaseline (SVD) Evaluation Results:")
    logger.info("-" * 50)

    for k in k_values:
        final_metrics[k] = {
            "recall": np.mean(results[k]["recall"]),
            "ndcg": np.mean(results[k]["ndcg"]),
            "hit_ratio": np.mean(results[k]["hit_ratio"]),
        }
        logger.info(
            f"@{k:2d}: Recall={final_metrics[k]['recall']:.4f}, "
            f"NDCG={final_metrics[k]['ndcg']:.4f}, "
            f"Hit Ratio={final_metrics[k]['hit_ratio']:.4f}"
        )

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline model")
    parser.add_argument(
        "--data", default=str(config.DATA_DIR), help="Directory containing dataset"
    )
    parser.add_argument(
        "--components", type=int, default=50, help="Number of latent components"
    )

    args = parser.parse_args()

    evaluate_baseline(args.data, args.components)


if __name__ == "__main__":
    main()
