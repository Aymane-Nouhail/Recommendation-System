"""
Dataset building utilities for recommendation system.

This module handles creating user-item interaction matrices and train/validation/test splits
using leave-one-out methodology for evaluation.
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builder class for creating recommendation datasets with proper train/val/test splits.
    """

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

    def fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit label encoders for users and items.

        Args:
            df: DataFrame containing user_id and asin columns
        """
        logger.info("Fitting label encoders for users and items")

        # Fit encoders
        self.user_encoder.fit(df["user_id"])
        self.item_encoder.fit(df["asin"])

        # Create mapping dictionaries for faster lookup
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_encoder.classes_)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_encoder.classes_)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        logger.info(f"Encoded {len(self.user_to_idx)} users and {len(self.item_to_idx)} items")

    def create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """
        Create user-item interaction matrix from DataFrame.

        Args:
            df: DataFrame with user_id, asin, and binary_rating columns

        Returns:
            Sparse interaction matrix (users x items)
        """
        logger.info("Creating user-item interaction matrix")

        # Encode users and items
        user_indices = df["user_id"].map(self.user_to_idx)
        item_indices = df["asin"].map(self.item_to_idx)

        # Handle any unmapped values (should not happen if encoders are fitted properly)
        valid_mask = user_indices.notna() & item_indices.notna()
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} interactions with unmapped users/items")
            df = df[valid_mask]
            user_indices = user_indices[valid_mask]
            item_indices = item_indices[valid_mask]

        # Create sparse matrix
        ratings = df["binary_rating"].values
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"Created interaction matrix of shape {interaction_matrix.shape}")
        logger.info(f"Matrix density: {interaction_matrix.nnz / (n_users * n_items):.6f}")

        return interaction_matrix

    def leave_one_out_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create leave-one-out train/validation/test splits.

        For each user:
        - Test set: Most recent interaction
        - Validation set: Second most recent interaction
        - Training set: All other interactions

        Optimized using vectorized operations instead of per-user loops.

        Args:
            df: DataFrame with user interactions

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating leave-one-out train/validation/test splits (optimized)")

        # Sort by timestamp to get chronological order
        df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Add reverse rank within each user group (last item = rank 1, second to last = rank 2, etc.)
        df_sorted["_rank"] = df_sorted.groupby("user_id").cumcount(ascending=False) + 1

        # Count interactions per user
        user_counts = df_sorted.groupby("user_id").size()
        df_sorted["_user_count"] = df_sorted["user_id"].map(user_counts)

        # Vectorized split logic:
        # - Test: rank == 1 (most recent) AND user has >= 2 interactions
        # - Val: rank == 2 (second most recent) AND user has >= 3 interactions
        # - Train: everything else

        test_mask = (df_sorted["_rank"] == 1) & (df_sorted["_user_count"] >= 2)
        val_mask = (df_sorted["_rank"] == 2) & (df_sorted["_user_count"] >= 3)
        train_mask = ~test_mask & ~val_mask

        # Extract splits and drop helper columns
        train_df = (
            df_sorted[train_mask].drop(columns=["_rank", "_user_count"]).reset_index(drop=True)
        )
        val_df = df_sorted[val_mask].drop(columns=["_rank", "_user_count"]).reset_index(drop=True)
        test_df = df_sorted[test_mask].drop(columns=["_rank", "_user_count"]).reset_index(drop=True)

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        logger.info(f"Users with validation data: {val_df['user_id'].nunique()}")
        logger.info(f"Users with test data: {test_df['user_id'].nunique()}")

        return train_df, val_df, test_df

    def create_negative_samples(
        self,
        df: pd.DataFrame,
        interaction_matrix: csr_matrix,
        n_negatives_per_positive: int = 4,
    ) -> pd.DataFrame:
        """
        Create negative samples for training (items not interacted with by users).

        Optimized version using vectorized NumPy operations and pre-computed lookups.

        Args:
            df: Positive interactions DataFrame
            interaction_matrix: User-item interaction matrix
            n_negatives_per_positive: Number of negative samples per positive

        Returns:
            DataFrame with negative samples added
        """
        logger.info(
            f"Creating negative samples ({n_negatives_per_positive} per positive) - optimized"
        )

        n_items = len(self.item_to_idx)
        all_items = np.arange(n_items)

        # Pre-compute: count positives per user (vectorized)
        user_positive_counts = df.groupby("user_id").size().to_dict()
        unique_users = list(user_positive_counts.keys())

        logger.info(f"Processing {len(unique_users)} users...")

        # Pre-compute user interaction sets from sparse matrix (much faster than DataFrame filtering)
        user_item_sets = {}
        for user_id in unique_users:
            user_idx = self.user_to_idx[user_id]
            user_item_sets[user_id] = set(interaction_matrix[user_idx].indices)

        # Batch sample all negatives
        negative_data = {
            "user_id": [],
            "asin": [],
            "binary_rating": [],
            "timestamp": [],
            "rating": [],
            "title": [],
            "text": [],
            "item_text": [],
        }

        for user_id in unique_users:
            user_items = user_item_sets[user_id]
            n_positives = user_positive_counts[user_id]
            n_negatives = n_positives * n_negatives_per_positive

            # Fast set difference using NumPy mask
            mask = np.ones(n_items, dtype=bool)
            mask[list(user_items)] = False
            candidate_items = all_items[mask]

            if len(candidate_items) == 0:
                continue

            # Sample negative items
            replace = len(candidate_items) < n_negatives
            negative_item_indices = np.random.choice(
                candidate_items, size=n_negatives, replace=replace
            )

            # Batch append (much faster than individual appends)
            negative_data["user_id"].extend([user_id] * n_negatives)
            negative_data["asin"].extend([self.idx_to_item[idx] for idx in negative_item_indices])
            negative_data["binary_rating"].extend([0] * n_negatives)
            negative_data["timestamp"].extend([0] * n_negatives)
            negative_data["rating"].extend([0] * n_negatives)
            negative_data["title"].extend([""] * n_negatives)
            negative_data["text"].extend([""] * n_negatives)
            negative_data["item_text"].extend([""] * n_negatives)

        # Create DataFrame from pre-built lists (single allocation)
        negatives_df = pd.DataFrame(negative_data)

        # Combine positive and negative samples
        combined_df = pd.concat([df, negatives_df], ignore_index=True)

        logger.info(f"Added {len(negatives_df)} negative samples")
        logger.info(f"Combined dataset size: {len(combined_df)}")

        return combined_df

    def save_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        interaction_matrix: csr_matrix,
        output_dir: str,
    ) -> None:
        """
        Save all dataset components to files.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            interaction_matrix: Full interaction matrix
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dataset to {output_path}")

        # Save DataFrames
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)

        # Save interaction matrix
        with open(output_path / "interaction_matrix.pkl", "wb") as f:
            pickle.dump(interaction_matrix, f)

        # Save encoders and mappings
        mappings = {
            "user_encoder": self.user_encoder,
            "item_encoder": self.item_encoder,
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
            "idx_to_user": self.idx_to_user,
            "idx_to_item": self.idx_to_item,
        }

        with open(output_path / "mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)

        # Save dataset statistics
        stats = {
            "n_users": len(self.user_to_idx),
            "n_items": len(self.item_to_idx),
            "train_interactions": len(train_df),
            "val_interactions": len(val_df),
            "test_interactions": len(test_df),
            "total_interactions": len(train_df) + len(val_df) + len(test_df),
            "matrix_density": interaction_matrix.nnz
            / (len(self.user_to_idx) * len(self.item_to_idx)),
            "train_users": train_df["user_id"].nunique() if not train_df.empty else 0,
            "val_users": val_df["user_id"].nunique() if not val_df.empty else 0,
            "test_users": test_df["user_id"].nunique() if not test_df.empty else 0,
        }

        with open(output_path / "dataset_stats.pkl", "wb") as f:
            pickle.dump(stats, f)

        # Print summary
        logger.info("Dataset saved successfully!")
        logger.info(f"  Training interactions: {stats['train_interactions']:,}")
        logger.info(f"  Validation interactions: {stats['val_interactions']:,}")
        logger.info(f"  Test interactions: {stats['test_interactions']:,}")
        logger.info(f"  Total users: {stats['n_users']:,}")
        logger.info(f"  Total items: {stats['n_items']:,}")
        logger.info(f"  Matrix density: {stats['matrix_density']:.6f}")


def build_recommendation_dataset(
    input_path: str,
    output_dir: str,
    add_negatives: bool = True,
    n_negatives_per_positive: int = 4,
) -> None:
    """
    Main function to build recommendation dataset with train/val/test splits.

    Args:
        input_path: Path to preprocessed data file (CSV or JSONL)
        output_dir: Directory to save processed dataset
        add_negatives: Whether to add negative samples to training data
        n_negatives_per_positive: Number of negative samples per positive
    """
    # Load preprocessed data
    logger.info(f"Loading preprocessed data from {input_path}")

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Input file must be CSV or JSONL format")

    # Initialize dataset builder
    builder = DatasetBuilder()

    # Fit encoders on full dataset
    builder.fit_encoders(df)

    # Create full interaction matrix
    interaction_matrix = builder.create_interaction_matrix(df)

    # Create train/val/test splits using leave-one-out
    train_df, val_df, test_df = builder.leave_one_out_split(df)

    # Add negative samples to training set if requested
    if add_negatives and not train_df.empty:
        train_df = builder.create_negative_samples(
            train_df, interaction_matrix, n_negatives_per_positive
        )

    # Save all components
    builder.save_dataset(train_df, val_df, test_df, interaction_matrix, output_dir)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Build recommendation dataset")
    parser.add_argument(
        "--input",
        default=config.PROCESSED_DATA_FILE,
        help="Input preprocessed data file (CSV or JSONL)",
    )
    parser.add_argument(
        "--output",
        default=str(config.DATA_DIR / "processed_dataset"),
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--no-negatives",
        action="store_true",
        help="Skip adding negative samples to training set",
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=4,
        help="Number of negative samples per positive (default: 4)",
    )

    args = parser.parse_args()

    build_recommendation_dataset(
        input_path=args.input,
        output_dir=args.output,
        add_negatives=not args.no_negatives,
        n_negatives_per_positive=args.n_negatives,
    )


if __name__ == "__main__":
    main()
