"""
Compute item text embeddings using Sentence-BERT (SBERT).

This module handles encoding item text (title + review text) into dense vector
representations using the pre-trained SBERT model "all-MiniLM-L6-v2".
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemEmbeddingGenerator:
    """
    Generator for creating item text embeddings using SBERT.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the SBERT model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Initializing SBERT model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Preprocess text for better embedding quality.

        Args:
            texts: List of raw text strings

        Returns:
            List of preprocessed text strings
        """
        processed_texts = []

        for text in texts:
            if pd.isna(text) or text == "":
                # Handle empty text
                processed_text = "[NO TEXT]"
            else:
                # Basic text cleaning
                processed_text = str(text).strip()

                # Remove excessive whitespace
                processed_text = " ".join(processed_text.split())

                # Truncate very long texts to prevent memory issues
                # SBERT can handle long texts, but we'll limit to 512 tokens worth of characters
                max_chars = 2048  # Roughly 512 tokens
                if len(processed_text) > max_chars:
                    processed_text = processed_text[:max_chars] + "..."

            processed_texts.append(processed_text)

        return processed_texts

    def encode_texts(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)
        """
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")

        # Preprocess texts
        processed_texts = self.preprocess_text(texts)

        # Encode in batches to handle memory constraints
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize embeddings
        )

        logger.info(f"Generated embeddings of shape: {embeddings.shape}")
        return embeddings

    def create_item_embeddings(
        self,
        df: pd.DataFrame,
        text_column: str = "item_text",
        item_id_column: str = "asin",
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Create embeddings for unique items in the dataset.

        Args:
            df: DataFrame containing item texts
            text_column: Column name containing item text
            item_id_column: Column name containing item IDs
            batch_size: Batch size for encoding

        Returns:
            Tuple of (embeddings_matrix, item_to_idx, idx_to_item)
        """
        logger.info("Creating item embeddings from dataset")

        # Get unique items and their texts
        unique_items = df[[item_id_column, text_column]].drop_duplicates(
            subset=[item_id_column]
        )
        unique_items = unique_items.sort_values(item_id_column).reset_index(drop=True)

        logger.info(f"Found {len(unique_items)} unique items")

        # Create item mappings
        item_to_idx = {
            item: idx for idx, item in enumerate(unique_items[item_id_column])
        }
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        # Extract texts for encoding
        item_texts = unique_items[text_column].tolist()

        # Generate embeddings
        embeddings = self.encode_texts(item_texts, batch_size=batch_size)

        # Validate embeddings
        assert len(embeddings) == len(
            unique_items
        ), "Mismatch in embeddings and items count"
        assert (
            embeddings.shape[1] == self.embedding_dim
        ), "Unexpected embedding dimension"

        logger.info(f"Created embedding matrix of shape: {embeddings.shape}")

        return embeddings, item_to_idx, idx_to_item

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        item_to_idx: Dict[str, int],
        idx_to_item: Dict[int, str],
        output_path: str,
        save_mappings: bool = True,
    ) -> None:
        """
        Save embeddings and mappings to files.

        Args:
            embeddings: Embedding matrix
            item_to_idx: Item ID to index mapping
            idx_to_item: Index to item ID mapping
            output_path: Output file path
            save_mappings: Whether to save mapping dictionaries
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving embeddings to {output_path}")

        # Save embeddings as numpy array
        np.save(output_path, embeddings)

        if save_mappings:
            # Save mappings
            mappings_path = output_path.with_name(output_path.stem + "_mappings.pkl")
            mappings = {
                "item_to_idx": item_to_idx,
                "idx_to_item": idx_to_item,
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "n_items": len(item_to_idx),
            }

            with open(mappings_path, "wb") as f:
                pickle.dump(mappings, f)

            logger.info(f"Saved mappings to {mappings_path}")

        # Save metadata
        metadata_path = output_path.with_name(output_path.stem + "_metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Embedding Model: {self.model_name}\n")
            f.write(f"Embedding Dimension: {self.embedding_dim}\n")
            f.write(f"Number of Items: {len(item_to_idx)}\n")
            f.write(f"Embedding Shape: {embeddings.shape}\n")
            f.write(f"Device Used: {self.device}\n")

        logger.info(f"Saved metadata to {metadata_path}")


def load_embeddings(
    embeddings_path: str, mappings_path: Optional[str] = None
) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
    """
    Load pre-computed embeddings and mappings.

    Args:
        embeddings_path: Path to embeddings numpy file
        mappings_path: Path to mappings pickle file (optional)

    Returns:
        Tuple of (embeddings, item_to_idx, idx_to_item)
    """
    logger.info(f"Loading embeddings from {embeddings_path}")

    # Load embeddings
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings of shape: {embeddings.shape}")

    # Load mappings if provided
    item_to_idx, idx_to_item = None, None
    if mappings_path and Path(mappings_path).exists():
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
        item_to_idx = mappings.get("item_to_idx")
        idx_to_item = mappings.get("idx_to_item")
        logger.info(f"Loaded mappings for {len(item_to_idx)} items")

    return embeddings, item_to_idx, idx_to_item


def compute_item_embeddings(
    input_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    text_column: str = "item_text",
    item_id_column: str = "asin",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> None:
    """
    Main function to compute and save item embeddings.

    Args:
        input_path: Path to input data file (CSV or JSONL)
        output_path: Path to save embeddings
        model_name: SBERT model name
        text_column: Column containing item text
        item_id_column: Column containing item IDs
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None)
    """
    # Load data
    logger.info(f"Loading data from {input_path}")

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Input file must be CSV or JSONL format")

    logger.info(f"Loaded {len(df)} records")

    # Validate required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    if item_id_column not in df.columns:
        raise ValueError(f"Item ID column '{item_id_column}' not found in data")

    # Initialize embedding generator
    generator = ItemEmbeddingGenerator(model_name=model_name, device=device)

    # Create embeddings
    embeddings, item_to_idx, idx_to_item = generator.create_item_embeddings(
        df=df,
        text_column=text_column,
        item_id_column=item_id_column,
        batch_size=batch_size,
    )

    # Save embeddings
    generator.save_embeddings(embeddings, item_to_idx, idx_to_item, output_path)

    logger.info("Item embeddings computation completed successfully!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compute item text embeddings using SBERT"
    )
    parser.add_argument(
        "--input",
        default=config.PROCESSED_DATA_FILE,
        help="Input data file (CSV or JSONL)",
    )
    parser.add_argument(
        "--output",
        default=config.EMBEDDINGS_FILE,
        help="Output path for embeddings (.npy file)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SBERT model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--text-column",
        default="item_text",
        help="Column name containing item text (default: item_text)",
    )
    parser.add_argument(
        "--item-column",
        default="asin",
        help="Column name containing item IDs (default: asin)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    compute_item_embeddings(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        text_column=args.text_column,
        item_id_column=args.item_column,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
