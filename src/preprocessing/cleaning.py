"""
Data preprocessing module for Amazon Reviews dataset.

Handles loading, cleaning, and preprocessing of JSONL Amazon Reviews data.
Creates item_text field by combining title and review text for SBERT embedding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str | Path) -> pd.DataFrame:
    """Load JSONL file into a DataFrame, skipping malformed lines."""
    logger.info(f"Loading JSONL from {file_path}")

    records = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading JSONL")):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {i + 1}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} records")
    return df


def validate_and_select_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields exist and select them."""
    required = ["user_id", "asin", "rating", "title", "text", "timestamp"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    logger.info("All required fields present")
    return df[required].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing critical fields, fill text fields."""
    initial_size = len(df)

    # Drop rows missing critical fields
    df = df.dropna(subset=["user_id", "asin", "rating"])

    # Fill missing text with empty string
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    logger.info(
        f"Cleaned: {initial_size:,} -> {len(df):,} rows ({initial_size - len(df):,} removed)"
    )
    return df


def add_derived_fields(df: pd.DataFrame, rating_threshold: float = 4.0) -> pd.DataFrame:
    """Add item_text and binary_rating fields."""
    # Combine title + text for embeddings
    df["item_text"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    df["item_text"] = df["item_text"].str.replace(r"\s+", " ", regex=True)

    # Binary rating (positive if >= threshold)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["binary_rating"] = (df["rating"] >= rating_threshold).astype(int)

    logger.info(f"Positive ratio: {df['binary_rating'].mean():.1%}")
    logger.info(f"Avg item_text length: {df['item_text'].str.len().mean():.0f} chars")
    return df


def filter_by_interactions(
    df: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
    max_iterations: int = 10,
) -> pd.DataFrame:
    """Iteratively filter users/items with insufficient interactions."""
    logger.info(f"Filtering (min_user={min_user}, min_item={min_item})")

    for i in range(max_iterations):
        prev_size = len(df)

        # Filter users
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]

        # Filter items
        item_counts = df["asin"].value_counts()
        df = df[df["asin"].isin(item_counts[item_counts >= min_item].index)]

        if len(df) == prev_size:
            logger.info(f"Converged after {i + 1} iterations")
            break

        logger.debug(f"Iter {i + 1}: {len(df):,} interactions")

    n_users, n_items = df["user_id"].nunique(), df["asin"].nunique()
    sparsity = 1 - len(df) / (n_users * n_items) if n_users * n_items > 0 else 0

    logger.info(f"Final: {len(df):,} interactions, {n_users:,} users, {n_items:,} items")
    logger.info(f"Sparsity: {sparsity:.4f}")
    return df


def preprocess_amazon_reviews(
    input_path: str | Path,
    output_path: str | Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    rating_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for Amazon Reviews dataset.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to save processed data (without extension)
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        rating_threshold: Threshold for binary rating conversion

    Returns:
        Processed DataFrame
    """
    df = load_jsonl(input_path)
    df = validate_and_select_fields(df)
    df = clean_data(df)
    df = add_derived_fields(df, rating_threshold)
    df = filter_by_interactions(df, min_user_interactions, min_item_interactions)

    # Save outputs
    output_path = Path(output_path)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    df.to_json(output_path.with_suffix(".jsonl"), orient="records", lines=True)

    logger.info(
        f"Saved to {output_path.with_suffix('.csv')} and {output_path.with_suffix('.jsonl')}"
    )
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews dataset")
    parser.add_argument("--input", default=config.RAW_DATA_FILE, help="Input JSONL file")
    parser.add_argument("--output", default=config.PROCESSED_DATA_FILE, help="Output file path")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--rating-threshold", type=float, default=4.0)

    args = parser.parse_args()

    preprocess_amazon_reviews(
        input_path=args.input,
        output_path=args.output,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        rating_threshold=args.rating_threshold,
    )


if __name__ == "__main__":
    main()
