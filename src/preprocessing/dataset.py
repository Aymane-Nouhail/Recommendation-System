"""
Dataset building utilities for recommendation system.

Creates user-item interaction matrices and train/val/test splits
using leave-one-out methodology for evaluation.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builder for recommendation datasets with train/val/test splits."""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.idx_to_item: dict[int, str] = {}

    def fit_encoders(self, df: pd.DataFrame) -> None:
        """Fit label encoders for users and items."""
        self.user_encoder.fit(df["user_id"])
        self.item_encoder.fit(df["asin"])

        self.user_to_idx = {u: i for i, u in enumerate(self.user_encoder.classes_)}
        self.item_to_idx = {it: i for i, it in enumerate(self.item_encoder.classes_)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        logger.info(f"Encoded {len(self.user_to_idx)} users, {len(self.item_to_idx)} items")

    def create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Create sparse user-item interaction matrix."""
        user_idx = df["user_id"].map(self.user_to_idx)
        item_idx = df["asin"].map(self.item_to_idx)

        valid = user_idx.notna() & item_idx.notna()
        if not valid.all():
            logger.warning(f"Dropping {(~valid).sum()} unmapped interactions")
            user_idx, item_idx = user_idx[valid], item_idx[valid]
            df = df[valid]

        n_users, n_items = len(self.user_to_idx), len(self.item_to_idx)
        matrix = csr_matrix(
            (df["binary_rating"].values, (user_idx, item_idx)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        logger.info(
            f"Matrix shape: {matrix.shape}, density: {matrix.nnz / (n_users * n_items):.6f}"
        )
        return matrix

    def leave_one_out_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-out split: test=most recent, val=2nd most recent, train=rest.
        """
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df["_rank"] = df.groupby("user_id").cumcount(ascending=False) + 1
        df["_count"] = df["user_id"].map(df.groupby("user_id").size())

        test_mask = (df["_rank"] == 1) & (df["_count"] >= 2)
        val_mask = (df["_rank"] == 2) & (df["_count"] >= 3)
        train_mask = ~test_mask & ~val_mask

        drop_cols = ["_rank", "_count"]
        train_df = df[train_mask].drop(columns=drop_cols).reset_index(drop=True)
        val_df = df[val_mask].drop(columns=drop_cols).reset_index(drop=True)
        test_df = df[test_mask].drop(columns=drop_cols).reset_index(drop=True)

        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    def create_negative_samples(
        self,
        df: pd.DataFrame,
        interaction_matrix: csr_matrix,
        n_negatives_per_positive: int = 4,
    ) -> pd.DataFrame:
        """Create negative samples (items user hasn't interacted with)."""
        n_items = len(self.item_to_idx)
        all_items = np.arange(n_items)

        user_counts = df.groupby("user_id").size().to_dict()
        users = list(user_counts.keys())

        # Pre-compute user interaction sets from sparse matrix
        user_items = {uid: set(interaction_matrix[self.user_to_idx[uid]].indices) for uid in users}

        neg_data = {col: [] for col in df.columns}

        for uid in users:
            positives = user_items[uid]
            n_neg = user_counts[uid] * n_negatives_per_positive

            # Sample from items user hasn't seen
            mask = np.ones(n_items, dtype=bool)
            mask[list(positives)] = False
            candidates = all_items[mask]

            if len(candidates) == 0:
                continue

            sampled = np.random.choice(candidates, size=n_neg, replace=len(candidates) < n_neg)

            neg_data["user_id"].extend([uid] * n_neg)
            neg_data["asin"].extend([self.idx_to_item[i] for i in sampled])
            neg_data["binary_rating"].extend([0] * n_neg)
            for col in df.columns:
                if col not in ["user_id", "asin", "binary_rating"]:
                    neg_data[col].extend([0 if col in ["timestamp", "rating"] else ""] * n_neg)

        neg_df = pd.DataFrame(neg_data)
        combined = pd.concat([df, neg_df], ignore_index=True)
        logger.info(f"Added {len(neg_df)} negatives, total: {len(combined)}")
        return combined

    def save_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        interaction_matrix: csr_matrix,
        output_dir: str | Path,
    ) -> None:
        """Save all dataset components."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(out / "val.csv", index=False)
        test_df.to_csv(out / "test.csv", index=False)

        with open(out / "interaction_matrix.pkl", "wb") as f:
            pickle.dump(interaction_matrix, f)

        mappings = {
            "user_encoder": self.user_encoder,
            "item_encoder": self.item_encoder,
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
            "idx_to_user": self.idx_to_user,
            "idx_to_item": self.idx_to_item,
        }
        with open(out / "mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)

        stats = {
            "n_users": len(self.user_to_idx),
            "n_items": len(self.item_to_idx),
            "train_interactions": len(train_df),
            "val_interactions": len(val_df),
            "test_interactions": len(test_df),
            "matrix_density": interaction_matrix.nnz
            / (len(self.user_to_idx) * len(self.item_to_idx)),
        }
        with open(out / "dataset_stats.pkl", "wb") as f:
            pickle.dump(stats, f)

        logger.info(f"Saved to {out}: {stats['n_users']} users, {stats['n_items']} items")


def build_recommendation_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    add_negatives: bool = True,
    n_negatives_per_positive: int = 4,
) -> None:
    """Build recommendation dataset with train/val/test splits."""
    logger.info(f"Loading data from {input_path}")

    path = str(input_path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input must be CSV or JSONL")

    builder = DatasetBuilder()
    builder.fit_encoders(df)
    matrix = builder.create_interaction_matrix(df)
    train_df, val_df, test_df = builder.leave_one_out_split(df)

    if add_negatives and not train_df.empty:
        train_df = builder.create_negative_samples(train_df, matrix, n_negatives_per_positive)

    builder.save_dataset(train_df, val_df, test_df, matrix, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build recommendation dataset")
    parser.add_argument("--input", default=config.PROCESSED_DATA_FILE)
    parser.add_argument("--output", default=str(config.DATA_DIR / "processed_dataset"))
    parser.add_argument("--no-negatives", action="store_true")
    parser.add_argument("--n-negatives", type=int, default=4)

    args = parser.parse_args()
    build_recommendation_dataset(
        input_path=args.input,
        output_dir=args.output,
        add_negatives=not args.no_negatives,
        n_negatives_per_positive=args.n_negatives,
    )


if __name__ == "__main__":
    main()
