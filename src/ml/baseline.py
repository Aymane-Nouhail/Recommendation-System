"""
Baseline models for recommendation system comparison.

This module implements multiple baseline recommenders:
- Popularity: Non-personalized, recommends most popular items
- Item-KNN: Item-based collaborative filtering with cosine similarity
- SVD: Matrix Factorization using TruncatedSVD
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.evaluate import hit_ratio_at_k, ndcg_at_k, recall_at_k
from ml.train import load_training_data
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Popularity Baseline
# =============================================================================


class PopularityRecommender:
    """
    Non-personalized recommender that ranks items by popularity.
    """

    def __init__(self):
        self.item_popularity = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Compute item popularity scores.

        Args:
            interaction_matrix: Sparse user-item interaction matrix
        """
        logger.info("Computing item popularity scores...")
        # Sum interactions per item (column-wise sum)
        self.item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
        logger.info(f"Top item has {self.item_popularity.max():.0f} interactions")

    def predict(self, user_idx: int) -> np.ndarray:
        """
        Return popularity scores (same for all users).

        Args:
            user_idx: User index (ignored, non-personalized)

        Returns:
            Array of item popularity scores
        """
        return self.item_popularity.copy()


# =============================================================================
# Item-KNN Baseline
# =============================================================================


class ItemKNNRecommender:
    """
    Item-based collaborative filtering using cosine similarity.
    """

    def __init__(self, k: int = 50):
        """
        Initialize Item-KNN.

        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.item_similarity = None
        self.interaction_matrix = None

    def fit(self, interaction_matrix: csr_matrix):
        """
        Compute item-item similarity matrix.

        Args:
            interaction_matrix: Sparse user-item interaction matrix
        """
        logger.info(f"Computing item-item similarity (k={self.k})...")
        self.interaction_matrix = interaction_matrix

        # Compute cosine similarity between items (transpose to get item-user matrix)
        # For large matrices, compute in chunks to save memory
        n_items = interaction_matrix.shape[1]

        if n_items > 10000:
            logger.info("Large item set, computing similarity in batches...")
            # For very large matrices, use approximate method
            # Just store the sparse interaction matrix and compute on-the-fly
            self.item_similarity = None
        else:
            # Compute full similarity matrix
            item_matrix = interaction_matrix.T.toarray()  # items x users
            self.item_similarity = cosine_similarity(item_matrix)
            # Zero out self-similarity
            np.fill_diagonal(self.item_similarity, 0)

        logger.info("Item similarity computation complete")

    def predict(self, user_idx: int) -> np.ndarray:
        """
        Predict scores based on similar items the user has interacted with.

        Args:
            user_idx: User index

        Returns:
            Array of item scores
        """
        # Get user's interaction vector
        user_interactions = self.interaction_matrix[user_idx].toarray().flatten()
        interacted_items = np.where(user_interactions > 0)[0]

        if len(interacted_items) == 0:
            return np.zeros(self.interaction_matrix.shape[1])

        if self.item_similarity is not None:
            # Use precomputed similarity
            # Score = sum of similarities to items user has interacted with
            scores = self.item_similarity[:, interacted_items].sum(axis=1)
        else:
            # Compute on-the-fly for large matrices
            item_matrix = self.interaction_matrix.T  # items x users
            user_items_matrix = item_matrix[interacted_items]
            scores = cosine_similarity(item_matrix, user_items_matrix).sum(axis=1)

        return scores


# =============================================================================
# SVD Baseline
# =============================================================================


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
        logger.info(f"Explained variance ratio: {self.model.explained_variance_ratio_.sum():.4f}")

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


def evaluate_baseline(data_dir: str, n_components: int = 50, k_values: List[int] | None = None):
    """
    Train and evaluate baseline model.

    Args:
        data_dir: Directory containing processed dataset
        n_components: Number of latent factors
        k_values: List of K values for evaluation
    """
    if k_values is None:
        k_values = [5, 10, 20]

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
        val_df[val_df["binary_rating"] == 1] if "binary_rating" in val_df.columns else val_df
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
        relevant_indices = [item_to_idx[item] for item in user_test_items if item in item_to_idx]

        if not relevant_indices:
            continue

        # Get predictions
        scores = model.predict(user_idx)

        # Mask training items (set score to -inf)
        # We want to recommend NEW items, not ones already seen in train/val
        user_train_items = input_df[input_df["user_id"] == user_id]["asin"].values
        train_indices = [item_to_idx[item] for item in user_train_items if item in item_to_idx]
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


def evaluate_model(
    model,
    model_name: str,
    input_matrix: csr_matrix,
    input_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k_values: List[int] | None = None,
    n_negatives: Optional[int] = 99,
) -> Dict:
    """
    Generic evaluation function for any baseline model.
    Supports both full ranking and negative sampling protocols.

    Args:
        model: Trained recommender model with predict(user_idx) method
        model_name: Name of the model for logging
        input_matrix: Training interaction matrix
        input_df: Training DataFrame
        test_df: Test DataFrame
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        k_values: List of K values for evaluation
        n_negatives: Number of negative samples (None = full ranking)

    Returns:
        Dictionary with evaluation metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    protocol = f"negative sampling ({n_negatives})" if n_negatives else "full ranking"
    logger.info(f"Evaluating {model_name} with {protocol}...")

    results = {}
    for k in k_values:
        results[k] = {"recall": [], "ndcg": [], "hit_ratio": []}

    n_items = input_matrix.shape[1]
    all_items = np.arange(n_items)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        user_id = row["user_id"]
        test_item_id = row["asin"]

        if user_id not in user_to_idx or test_item_id not in item_to_idx:
            continue

        user_idx = user_to_idx[user_id]
        test_item_idx = item_to_idx[test_item_id]

        # Get items the user has seen (to exclude from candidates)
        seen_items = set(input_matrix[user_idx].indices)

        # Get all scores from model
        all_scores = model.predict(user_idx)

        if n_negatives is not None:
            # NEGATIVE SAMPLING PROTOCOL
            # Sample n_negatives random items that user hasn't seen
            candidate_mask = np.ones(n_items, dtype=bool)
            candidate_mask[list(seen_items)] = False
            candidate_mask[test_item_idx] = False  # Don't sample test item as negative

            available_negatives = all_items[candidate_mask]

            if len(available_negatives) < n_negatives:
                negative_items = available_negatives
            else:
                negative_items = np.random.choice(
                    available_negatives, size=n_negatives, replace=False
                )

            # Candidate set = 1 positive + n negatives
            candidate_items = np.concatenate([[test_item_idx], negative_items])

            # Get scores for candidates only and rank
            candidate_scores = all_scores[candidate_items]
            ranked_indices = np.argsort(candidate_scores)[::-1]
            ranked_items = candidate_items[ranked_indices]
        else:
            # FULL RANKING PROTOCOL
            # Mask seen items
            all_scores[list(seen_items)] = -np.inf

            # Rank all items
            max_k = max(k_values)
            ranked_items = np.argsort(all_scores)[::-1][:max_k]

        # The test item is the relevant item
        relevant_indices = np.array([test_item_idx])

        # Calculate metrics
        for k in k_values:
            k_recs = ranked_items[:k]
            results[k]["recall"].append(recall_at_k(k_recs, relevant_indices, k))
            results[k]["ndcg"].append(ndcg_at_k(k_recs, relevant_indices, k))
            results[k]["hit_ratio"].append(hit_ratio_at_k(k_recs, relevant_indices, k))

    # Aggregate results
    final_metrics = {}
    for k in k_values:
        final_metrics[k] = {
            "recall": np.mean(results[k]["recall"]) if results[k]["recall"] else 0.0,
            "ndcg": np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else 0.0,
            "hit_ratio": np.mean(results[k]["hit_ratio"]) if results[k]["hit_ratio"] else 0.0,
        }

    return final_metrics


def run_all_baselines(
    data_dir: str, k_values: List[int] | None = None, n_negatives: Optional[int] = 99
):
    """
    Run all baseline models and compare results.

    Args:
        data_dir: Directory containing processed dataset
        k_values: List of K values for evaluation
        n_negatives: Number of negative samples (None = full ranking)
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # Load data
    full_interaction_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    # Reconstruct input matrix from train and val data only
    logger.info("Reconstructing input matrix from train/val data...")
    train_positives = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )
    val_positives = (
        val_df[val_df["binary_rating"] == 1] if "binary_rating" in val_df.columns else val_df
    )
    input_df = pd.concat([train_positives, val_positives])

    rows = input_df["user_id"].map(user_to_idx)
    cols = input_df["asin"].map(item_to_idx)
    data = np.ones(len(input_df))
    input_matrix = csr_matrix((data, (rows, cols)), shape=full_interaction_matrix.shape)

    # Load test data
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    # Store all results
    all_results = {}

    protocol = f"negative sampling ({n_negatives})" if n_negatives else "full ranking"
    logger.info(f"\nEvaluation protocol: {protocol}")

    # 1. Popularity Baseline
    logger.info("\n" + "=" * 60)
    logger.info("Training Popularity Baseline")
    logger.info("=" * 60)
    pop_model = PopularityRecommender()
    pop_model.fit(input_matrix)
    all_results["Popularity"] = evaluate_model(
        pop_model,
        "Popularity",
        input_matrix,
        input_df,
        test_df,
        user_to_idx,
        item_to_idx,
        k_values,
        n_negatives,
    )

    # 2. Item-KNN Baseline
    logger.info("\n" + "=" * 60)
    logger.info("Training Item-KNN Baseline")
    logger.info("=" * 60)
    knn_model = ItemKNNRecommender(k=50)
    knn_model.fit(input_matrix)
    all_results["Item-KNN"] = evaluate_model(
        knn_model,
        "Item-KNN",
        input_matrix,
        input_df,
        test_df,
        user_to_idx,
        item_to_idx,
        k_values,
        n_negatives,
    )

    # 3. SVD Baseline
    logger.info("\n" + "=" * 60)
    logger.info("Training SVD Baseline")
    logger.info("=" * 60)
    svd_model = SVDRecommender(n_components=50)
    svd_model.fit(input_matrix)
    all_results["SVD"] = evaluate_model(
        svd_model,
        "SVD",
        input_matrix,
        input_df,
        test_df,
        user_to_idx,
        item_to_idx,
        k_values,
        n_negatives,
    )

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info(f"Protocol: {protocol}")
    logger.info("=" * 70)

    print("\n" + "-" * 75)
    print(
        f"{'Model':<15} | {'Recall@5':>10} | {'Recall@10':>10} | {'Recall@20':>10} | {'NDCG@20':>10}"
    )
    print("-" * 75)

    for model_name, metrics in all_results.items():
        print(
            f"{model_name:<15} | {metrics[5]['recall']:>10.4f} | {metrics[10]['recall']:>10.4f} | "
            f"{metrics[20]['recall']:>10.4f} | {metrics[20]['ndcg']:>10.4f}"
        )

    print("-" * 75)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="Directory containing dataset")
    parser.add_argument(
        "--model",
        choices=["all", "popularity", "itemknn", "svd"],
        default="all",
        help="Which baseline to run (default: all)",
    )
    parser.add_argument(
        "--components", type=int, default=50, help="Number of latent components for SVD"
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=99,
        help="Number of negative samples for evaluation (default: 99). Use 0 for full ranking.",
    )

    args = parser.parse_args()

    # Determine evaluation protocol
    n_negatives = args.n_negatives if args.n_negatives > 0 else None

    if args.model == "all":
        run_all_baselines(args.data, n_negatives=n_negatives)
    else:
        # Run individual baseline (uses existing evaluate_baseline for SVD)
        evaluate_baseline(args.data, args.components)


if __name__ == "__main__":
    main()
