"""
Dataset building utilities for recommendation system.

This module handles creating user-item interaction matrices and train/validation/test splits
using leave-one-out methodology for evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Set
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import logging
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
        self.user_encoder.fit(df['user_id'])
        self.item_encoder.fit(df['asin'])
        
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
        user_indices = df['user_id'].map(self.user_to_idx)
        item_indices = df['asin'].map(self.item_to_idx)
        
        # Handle any unmapped values (should not happen if encoders are fitted properly)
        valid_mask = user_indices.notna() & item_indices.notna()
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} interactions with unmapped users/items")
            df = df[valid_mask]
            user_indices = user_indices[valid_mask]
            item_indices = item_indices[valid_mask]
        
        # Create sparse matrix
        ratings = df['binary_rating'].values
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        logger.info(f"Created interaction matrix of shape {interaction_matrix.shape}")
        logger.info(f"Matrix density: {interaction_matrix.nnz / (n_users * n_items):.6f}")
        
        return interaction_matrix
    
    def leave_one_out_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create leave-one-out train/validation/test splits.
        
        For each user:
        - Test set: Most recent interaction
        - Validation set: Second most recent interaction  
        - Training set: All other interactions
        
        Args:
            df: DataFrame with user interactions
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating leave-one-out train/validation/test splits")
        
        # Sort by timestamp to get chronological order
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        
        train_data = []
        val_data = []
        test_data = []
        
        for user_id, user_df in df_sorted.groupby('user_id'):
            user_interactions = user_df.reset_index(drop=True)
            n_interactions = len(user_interactions)
            
            if n_interactions >= 3:
                # Split: train (all but last 2), val (second to last), test (last)
                train_data.append(user_interactions.iloc[:-2])
                val_data.append(user_interactions.iloc[-2:-1])
                test_data.append(user_interactions.iloc[-1:])
            elif n_interactions == 2:
                # Split: train (first), test (last), no validation
                train_data.append(user_interactions.iloc[:-1])
                test_data.append(user_interactions.iloc[-1:])
            else:
                # Only one interaction: put in training set
                train_data.append(user_interactions)
        
        # Combine all splits
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Users with validation data: {val_df['user_id'].nunique()}")
        logger.info(f"Users with test data: {test_df['user_id'].nunique()}")
        
        return train_df, val_df, test_df
    
    def create_negative_samples(self, 
                               df: pd.DataFrame, 
                               interaction_matrix: csr_matrix,
                               n_negatives_per_positive: int = 4) -> pd.DataFrame:
        """
        Create negative samples for training (items not interacted with by users).
        
        Args:
            df: Positive interactions DataFrame
            interaction_matrix: User-item interaction matrix
            n_negatives_per_positive: Number of negative samples per positive
            
        Returns:
            DataFrame with negative samples added
        """
        logger.info(f"Creating negative samples ({n_negatives_per_positive} per positive)")
        
        negative_samples = []
        n_items = len(self.item_to_idx)
        
        for user_id in df['user_id'].unique():
            user_idx = self.user_to_idx[user_id]
            
            # Get items this user has interacted with
            user_items = set(interaction_matrix[user_idx].nonzero()[1])
            
            # Get items this user has NOT interacted with
            all_items = set(range(n_items))
            candidate_items = list(all_items - user_items)
            
            if not candidate_items:
                continue
                
            # Count positive interactions for this user in current DataFrame
            n_positives = len(df[df['user_id'] == user_id])
            n_negatives = n_positives * n_negatives_per_positive
            
            # Sample negative items
            if len(candidate_items) >= n_negatives:
                negative_item_indices = np.random.choice(
                    candidate_items, size=n_negatives, replace=False
                )
            else:
                negative_item_indices = np.random.choice(
                    candidate_items, size=n_negatives, replace=True
                )
            
            # Convert back to item IDs
            for item_idx in negative_item_indices:
                item_id = self.idx_to_item[item_idx]
                negative_samples.append({
                    'user_id': user_id,
                    'asin': item_id,
                    'binary_rating': 0,
                    'timestamp': 0,  # Dummy timestamp
                    'rating': 0,     # Dummy rating
                    'title': '',     # Dummy title
                    'text': '',      # Dummy text
                    'item_text': ''  # Dummy item_text
                })
        
        # Create DataFrame with negative samples
        negatives_df = pd.DataFrame(negative_samples)
        
        # Combine positive and negative samples
        combined_df = pd.concat([df, negatives_df], ignore_index=True)
        
        logger.info(f"Added {len(negatives_df)} negative samples")
        logger.info(f"Combined dataset size: {len(combined_df)}")
        
        return combined_df
    
    def save_dataset(self, 
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame, 
                    test_df: pd.DataFrame,
                    interaction_matrix: csr_matrix,
                    output_dir: str) -> None:
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
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        # Save interaction matrix
        with open(output_path / 'interaction_matrix.pkl', 'wb') as f:
            pickle.dump(interaction_matrix, f)
        
        # Save encoders and mappings
        mappings = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item
        }
        
        with open(output_path / 'mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save dataset statistics
        stats = {
            'n_users': len(self.user_to_idx),
            'n_items': len(self.item_to_idx),
            'train_interactions': len(train_df),
            'val_interactions': len(val_df),
            'test_interactions': len(test_df),
            'total_interactions': len(train_df) + len(val_df) + len(test_df),
            'matrix_density': interaction_matrix.nnz / (len(self.user_to_idx) * len(self.item_to_idx)),
            'train_users': train_df['user_id'].nunique() if not train_df.empty else 0,
            'val_users': val_df['user_id'].nunique() if not val_df.empty else 0,
            'test_users': test_df['user_id'].nunique() if not test_df.empty else 0,
        }
        
        with open(output_path / 'dataset_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        
        # Print summary
        logger.info("Dataset saved successfully!")
        logger.info(f"  Training interactions: {stats['train_interactions']:,}")
        logger.info(f"  Validation interactions: {stats['val_interactions']:,}")
        logger.info(f"  Test interactions: {stats['test_interactions']:,}")
        logger.info(f"  Total users: {stats['n_users']:,}")
        logger.info(f"  Total items: {stats['n_items']:,}")
        logger.info(f"  Matrix density: {stats['matrix_density']:.6f}")


def build_recommendation_dataset(input_path: str,
                                output_dir: str,
                                add_negatives: bool = True,
                                n_negatives_per_positive: int = 4) -> None:
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
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.jsonl'):
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
    parser = argparse.ArgumentParser(description='Build recommendation dataset')
    parser.add_argument('--input', default=config.PROCESSED_DATA_FILE, 
                       help='Input preprocessed data file (CSV or JSONL)')
    parser.add_argument('--output', default=str(config.DATA_DIR / "processed_dataset"),
                       help='Output directory for dataset files')
    parser.add_argument('--no-negatives', action='store_true',
                       help='Skip adding negative samples to training set')
    parser.add_argument('--n-negatives', type=int, default=4,
                       help='Number of negative samples per positive (default: 4)')
    
    args = parser.parse_args()
    
    build_recommendation_dataset(
        input_path=args.input,
        output_dir=args.output,
        add_negatives=not args.no_negatives,
        n_negatives_per_positive=args.n_negatives
    )


if __name__ == "__main__":
    main()