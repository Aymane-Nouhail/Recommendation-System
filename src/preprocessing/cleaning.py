"""
Data preprocessing module for Amazon Reviews dataset.

This module handles loading, cleaning, and preprocessing of JSONL Amazon Reviews data.
It creates the item_text field by combining title and review text for SBERT embedding.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import logging
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> pd.DataFrame:
    """
    Load JSONL file into a pandas DataFrame.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading JSONL file from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading JSONL")):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num + 1}: {e}")
                continue
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records")
    return df


def validate_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that required fields exist in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validated fields
    """
    required_fields = ['user_id', 'asin', 'rating', 'title', 'text', 'timestamp']
    
    # Check if all required fields exist
    missing_fields = set(required_fields) - set(df.columns)
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    logger.info("All required fields present in dataset")
    return df[required_fields]


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Dataset shape before cleaning: {df.shape}")
    
    # Log missing values per column
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            logger.info(f"Missing values in {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Remove rows with missing critical fields
    critical_fields = ['user_id', 'asin', 'rating']
    df_clean = df.dropna(subset=critical_fields)
    
    # Fill missing text fields with empty strings
    text_fields = ['title', 'text']
    for field in text_fields:
        df_clean[field] = df_clean[field].fillna('')
    
    logger.info(f"Dataset shape after cleaning: {df_clean.shape}")
    logger.info(f"Removed {len(df) - len(df_clean)} rows with missing critical fields")
    
    return df_clean


def build_item_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build item_text field by combining title and review text.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added item_text column
    """
    logger.info("Building item_text field from title and review text")
    
    # Combine title and text with a separator
    df['item_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
    
    # Clean up extra whitespace
    df['item_text'] = df['item_text'].str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Log statistics
    avg_length = df['item_text'].str.len().mean()
    logger.info(f"Average item_text length: {avg_length:.1f} characters")
    
    return df


def filter_data(df: pd.DataFrame, 
                min_user_interactions: int = 5,
                min_item_interactions: int = 5) -> pd.DataFrame:
    """
    Filter data to remove users and items with too few interactions.
    
    Args:
        df: Input DataFrame
        min_user_interactions: Minimum number of interactions per user
        min_item_interactions: Minimum number of interactions per item
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering data (min_user: {min_user_interactions}, min_item: {min_item_interactions})")
    logger.info(f"Initial dataset size: {len(df)} interactions")
    
    # Iteratively filter users and items until convergence
    prev_size = 0
    iteration = 0
    
    while len(df) != prev_size and iteration < 10:
        prev_size = len(df)
        iteration += 1
        
        # Filter users with enough interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items with enough interactions
        item_counts = df['asin'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['asin'].isin(valid_items)]
        
        logger.info(f"Iteration {iteration}: {len(df)} interactions, "
                   f"{df['user_id'].nunique()} users, {df['asin'].nunique()} items")
    
    logger.info(f"Final filtered dataset: {len(df)} interactions, "
               f"{df['user_id'].nunique()} users, {df['asin'].nunique()} items")
    
    return df


def convert_ratings_to_binary(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Convert ratings to binary (1 for relevant, 0 for not relevant).
    
    Args:
        df: Input DataFrame
        threshold: Rating threshold (>= threshold -> 1, < threshold -> 0)
        
    Returns:
        DataFrame with binary ratings
    """
    logger.info(f"Converting ratings to binary with threshold {threshold}")
    
    # Convert rating to numeric if it's not already
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Remove rows with invalid ratings
    df = df.dropna(subset=['rating'])
    
    # Convert to binary
    df['binary_rating'] = (df['rating'] >= threshold).astype(int)
    
    # Log statistics
    positive_ratio = df['binary_rating'].mean()
    logger.info(f"Positive interaction ratio: {positive_ratio:.3f}")
    
    return df


def preprocess_amazon_reviews(input_path: str, 
                             output_path: str,
                             min_user_interactions: int = 5,
                             min_item_interactions: int = 5,
                             rating_threshold: float = 4.0) -> None:
    """
    Main preprocessing pipeline for Amazon Reviews dataset.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to save processed data
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        rating_threshold: Threshold for binary rating conversion
    """
    # Load raw data
    df = load_jsonl(input_path)
    
    # Validate required fields
    df = validate_required_fields(df)
    
    # Clean missing values
    df = clean_missing_values(df)
    
    # Build item text
    df = build_item_text(df)
    
    # Convert ratings to binary
    df = convert_ratings_to_binary(df, rating_threshold)
    
    # Filter data
    df = filter_data(df, min_user_interactions, min_item_interactions)
    
    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    
    # Save as both CSV and JSONL for flexibility
    output_path_path = Path(output_path)
    
    # Save as CSV
    csv_path = output_path_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Save as JSONL
    jsonl_path = output_path_path.with_suffix('.jsonl')
    df.to_json(jsonl_path, orient='records', lines=True)
    logger.info(f"Saved JSONL to {jsonl_path}")
    
    # Print final statistics
    logger.info(f"\nFinal Dataset Statistics:")
    logger.info(f"  Total interactions: {len(df):,}")
    logger.info(f"  Unique users: {df['user_id'].nunique():,}")
    logger.info(f"  Unique items: {df['asin'].nunique():,}")
    logger.info(f"  Positive interactions: {df['binary_rating'].sum():,}")
    logger.info(f"  Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['asin'].nunique()):.4f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Preprocess Amazon Reviews dataset')
    parser.add_argument('--input', default=config.RAW_DATA_FILE, help='Input JSONL file path')
    parser.add_argument('--output', default=config.PROCESSED_DATA_FILE, help='Output file path')
    parser.add_argument('--min-user-interactions', type=int, default=5,
                       help='Minimum interactions per user (default: 5)')
    parser.add_argument('--min-item-interactions', type=int, default=5,
                       help='Minimum interactions per item (default: 5)')
    parser.add_argument('--rating-threshold', type=float, default=4.0,
                       help='Rating threshold for binary conversion (default: 4.0)')
    
    args = parser.parse_args()
    
    preprocess_amazon_reviews(
        input_path=args.input,
        output_path=args.output,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        rating_threshold=args.rating_threshold
    )


if __name__ == "__main__":
    main()