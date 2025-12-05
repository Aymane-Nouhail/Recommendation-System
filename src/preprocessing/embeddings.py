"""
Compute item text embeddings using Sentence-BERT (SBERT).

Encodes item text (title + review) into dense vectors using "all-MiniLM-L6-v2".
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

MAX_CHARS = 2048  # ~512 tokens


class ItemEmbeddingGenerator:
    """Generator for item text embeddings using SBERT."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading SBERT model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _preprocess_text(self, texts: list[str]) -> list[str]:
        """Clean and truncate texts."""
        processed = []
        for text in texts:
            if pd.isna(text) or text == "":
                processed.append("[NO TEXT]")
            else:
                t = " ".join(str(text).split())[:MAX_CHARS]
                processed.append(t + "..." if len(str(text)) > MAX_CHARS else t)
        return processed

    def encode_texts(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")
        processed = self._preprocess_text(texts)

        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def create_item_embeddings(
        self,
        df: pd.DataFrame,
        text_column: str = "item_text",
        item_id_column: str = "asin",
        batch_size: int = 32,
    ) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
        """Create embeddings for unique items."""
        unique = (
            df[[item_id_column, text_column]]
            .drop_duplicates(subset=[item_id_column])
            .sort_values(item_id_column)
            .reset_index(drop=True)
        )
        logger.info(f"Found {len(unique)} unique items")

        item_to_idx = {item: i for i, item in enumerate(unique[item_id_column])}
        idx_to_item = {i: item for item, i in item_to_idx.items()}

        embeddings = self.encode_texts(unique[text_column].tolist(), batch_size)

        assert len(embeddings) == len(unique)
        assert embeddings.shape[1] == self.embedding_dim

        return embeddings, item_to_idx, idx_to_item

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        item_to_idx: dict[str, int],
        idx_to_item: dict[int, str],
        output_path: str | Path,
    ) -> None:
        """Save embeddings, mappings, and metadata."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(out, embeddings)
        logger.info(f"Saved embeddings to {out}")

        # Save mappings
        mappings_path = out.with_name(f"{out.stem}_mappings.pkl")
        with open(mappings_path, "wb") as f:
            pickle.dump(
                {
                    "item_to_idx": item_to_idx,
                    "idx_to_item": idx_to_item,
                    "embedding_dim": self.embedding_dim,
                    "model_name": self.model_name,
                    "n_items": len(item_to_idx),
                },
                f,
            )
        logger.info(f"Saved mappings to {mappings_path}")

        # Save metadata
        meta_path = out.with_name(f"{out.stem}_metadata.txt")
        meta_path.write_text(
            f"Model: {self.model_name}\n"
            f"Dimension: {self.embedding_dim}\n"
            f"Items: {len(item_to_idx)}\n"
            f"Shape: {embeddings.shape}\n"
            f"Device: {self.device}\n"
        )


def load_embeddings(
    embeddings_path: str | Path,
    mappings_path: str | Path | None = None,
) -> tuple[np.ndarray, dict | None, dict | None]:
    """Load pre-computed embeddings and optional mappings."""
    embeddings_path = Path(embeddings_path)
    logger.info(f"Loading embeddings from {embeddings_path}")

    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings of shape: {embeddings.shape}")

    item_to_idx, idx_to_item = None, None

    # Auto-detect mappings path if not provided
    if mappings_path is None:
        mappings_path = embeddings_path.with_name(f"{embeddings_path.stem}_mappings.pkl")

    if Path(mappings_path).exists():
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
        item_to_idx = mappings.get("item_to_idx")
        idx_to_item = mappings.get("idx_to_item")
        logger.info(f"Loaded mappings for {len(item_to_idx)} items")

    return embeddings, item_to_idx, idx_to_item


def compute_item_embeddings(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    text_column: str = "item_text",
    item_id_column: str = "asin",
    batch_size: int = 32,
    device: str | None = None,
) -> None:
    """Compute and save item embeddings from data file."""
    logger.info(f"Loading data from {input_path}")

    path = str(input_path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input must be CSV or JSONL")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")
    if item_id_column not in df.columns:
        raise ValueError(f"Column '{item_id_column}' not found")

    generator = ItemEmbeddingGenerator(model_name, device)
    embeddings, item_to_idx, idx_to_item = generator.create_item_embeddings(
        df, text_column, item_id_column, batch_size
    )
    generator.save_embeddings(embeddings, item_to_idx, idx_to_item, output_path)

    logger.info("Done!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute SBERT item embeddings")
    parser.add_argument("--input", default=config.PROCESSED_DATA_FILE)
    parser.add_argument("--output", default=config.EMBEDDINGS_FILE)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--text-column", default="item_text")
    parser.add_argument("--item-column", default="asin")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)

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
