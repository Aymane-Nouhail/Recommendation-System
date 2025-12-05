"""
Integration test for the full recommendation pipeline using mock data.

Tests the complete flow: data → embeddings → training → evaluation → inference
without requiring external models or large datasets.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from src.ml.model import HybridVAE, vae_loss_function
from src.ml.train import UserInteractionDataset, VAETrainer
from src.preprocessing.dataset import DatasetBuilder

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_interactions():
    """Create a small synthetic interaction dataset."""
    np.random.seed(42)

    # 20 users, 50 items, ~200 interactions
    n_users, n_items, n_interactions = 20, 50, 200

    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]

    data = []
    for _ in range(n_interactions):
        user = np.random.choice(users)
        item = np.random.choice(items)
        timestamp = np.random.randint(1600000000, 1700000000)
        rating = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
        data.append(
            {
                "user_id": user,
                "asin": item,
                "timestamp": timestamp,
                "rating": rating,
                "binary_rating": 1.0 if rating >= 3.0 else 0.0,
                "title": f"Title for {item}",
                "text": f"Review text for {item}",
                "item_text": f"Title for {item}. Review text for {item}",
            }
        )

    df = pd.DataFrame(data).drop_duplicates(subset=["user_id", "asin"])
    return df


@pytest.fixture
def mock_embeddings():
    """Create mock SBERT-like embeddings (384 dimensions like all-MiniLM-L6-v2)."""
    np.random.seed(42)
    n_items = 50
    embedding_dim = 384

    # Random normalized embeddings
    embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    item_to_idx = {f"item_{i}": i for i in range(n_items)}
    idx_to_item = {i: f"item_{i}" for i in range(n_items)}

    return embeddings, item_to_idx, idx_to_item


@pytest.fixture
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Unit Tests
# =============================================================================


class TestDatasetBuilder:
    """Tests for DatasetBuilder."""

    def test_fit_encoders(self, sample_interactions):
        """Test that encoders are fitted correctly."""
        builder = DatasetBuilder()
        builder.fit_encoders(sample_interactions)

        assert len(builder.user_to_idx) > 0
        assert len(builder.item_to_idx) > 0
        assert len(builder.user_to_idx) == len(builder.idx_to_user)
        assert len(builder.item_to_idx) == len(builder.idx_to_item)

    def test_create_interaction_matrix(self, sample_interactions):
        """Test interaction matrix creation."""
        builder = DatasetBuilder()
        builder.fit_encoders(sample_interactions)
        matrix = builder.create_interaction_matrix(sample_interactions)

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape[0] == len(builder.user_to_idx)
        assert matrix.shape[1] == len(builder.item_to_idx)
        assert matrix.nnz > 0

    def test_leave_one_out_split(self, sample_interactions):
        """Test train/val/test split."""
        builder = DatasetBuilder()
        train_df, val_df, test_df = builder.leave_one_out_split(sample_interactions)

        # All splits should have data
        assert len(train_df) > 0
        assert len(test_df) > 0
        # val_df might be empty if users don't have enough interactions

        # No overlap between splits
        train_keys = set(zip(train_df["user_id"], train_df["asin"]))
        test_keys = set(zip(test_df["user_id"], test_df["asin"]))
        assert len(train_keys & test_keys) == 0


class TestHybridVAE:
    """Tests for HybridVAE model."""

    def test_model_initialization(self, mock_embeddings, device):
        """Test model initializes correctly."""
        embeddings, _, _ = mock_embeddings
        n_items = embeddings.shape[0]

        model = HybridVAE(
            n_items=n_items,
            item_embeddings=embeddings,
            latent_dim=64,
            hidden_dims=[128],
            dropout=0.3,
            beta=0.2,
        )

        assert model.n_items == n_items
        assert model.latent_dim == 64
        assert model.embedding_dim == embeddings.shape[1]

    def test_forward_pass(self, mock_embeddings, device):
        """Test forward pass produces correct shapes."""
        embeddings, _, _ = mock_embeddings
        n_items = embeddings.shape[0]
        batch_size = 4

        model = HybridVAE(
            n_items=n_items,
            item_embeddings=embeddings,
            latent_dim=64,
            hidden_dims=[128],
        ).to(device)

        # Random input
        x = torch.randn(batch_size, n_items).to(device)
        recon_x, mu, logvar = model(x)

        assert recon_x.shape == (batch_size, n_items)
        assert mu.shape == (batch_size, 64)
        assert logvar.shape == (batch_size, 64)

    def test_encode_decode(self, mock_embeddings, device):
        """Test encode and decode methods."""
        embeddings, _, _ = mock_embeddings
        n_items = embeddings.shape[0]

        model = HybridVAE(
            n_items=n_items,
            item_embeddings=embeddings,
            latent_dim=64,
            hidden_dims=[128],
        ).to(device)

        x = torch.randn(2, n_items).to(device)

        # Test encoding
        mu, logvar = model.encode(x)
        assert mu.shape == (2, 64)

        # Test user embedding
        z = model.get_user_embedding(x)
        assert z.shape == (2, 64)

        # Test decoding
        recon = model.decode(z)
        assert recon.shape == (2, n_items)

    def test_loss_function(self, mock_embeddings, device):
        """Test VAE loss computation."""
        embeddings, _, _ = mock_embeddings
        n_items = embeddings.shape[0]

        model = HybridVAE(
            n_items=n_items,
            item_embeddings=embeddings,
            latent_dim=64,
            hidden_dims=[128],
            beta=0.2,
        ).to(device)

        x = torch.randn(4, n_items).to(device)
        recon_x, mu, logvar = model(x)

        loss, recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, logvar, beta=0.2)

        assert not torch.isnan(loss)
        assert not torch.isnan(recon_loss)
        assert kl_loss.item() >= 0  # KL divergence is always non-negative


class TestTraining:
    """Tests for training components."""

    def test_user_interaction_dataset(self, sample_interactions):
        """Test dataset creation."""
        builder = DatasetBuilder()
        builder.fit_encoders(sample_interactions)
        matrix = builder.create_interaction_matrix(sample_interactions)

        dataset = UserInteractionDataset(matrix)

        assert len(dataset) == matrix.shape[0]

        # Test __getitem__
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (matrix.shape[1],)

    def test_trainer_epoch(self, sample_interactions, mock_embeddings, device):
        """Test training for one epoch."""
        # Prepare data
        builder = DatasetBuilder()
        builder.fit_encoders(sample_interactions)
        matrix = builder.create_interaction_matrix(sample_interactions)

        embeddings, item_to_idx, _ = mock_embeddings

        # Align embeddings with dataset items
        n_items = len(builder.item_to_idx)
        aligned_embeddings = np.random.randn(n_items, 384).astype(np.float32)

        # Create model and trainer
        model = HybridVAE(
            n_items=n_items,
            item_embeddings=aligned_embeddings,
            latent_dim=64,
            hidden_dims=[128],
            beta=0.2,
        )

        trainer = VAETrainer(model, device, lr=0.001)

        # Create dataloader
        dataset = UserInteractionDataset(matrix)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Train one epoch
        metrics = trainer.train_epoch(loader)

        assert "total_loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics
        assert metrics["total_loss"] > 0


# =============================================================================
# Integration Test
# =============================================================================


class TestFullPipeline:
    """Integration test for the complete pipeline."""

    def test_end_to_end_pipeline(self, sample_interactions, device):
        """Test the full pipeline from data to recommendations."""
        # Step 1: Build dataset
        builder = DatasetBuilder()
        train_df, val_df, test_df = builder.leave_one_out_split(sample_interactions)
        builder.fit_encoders(train_df)
        train_matrix = builder.create_interaction_matrix(train_df)

        n_items = len(builder.item_to_idx)

        # Step 2: Mock embeddings (simulating SBERT output)
        np.random.seed(42)
        embeddings = np.random.randn(n_items, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Step 3: Create and train model
        model = HybridVAE(
            n_items=n_items,
            item_embeddings=embeddings,
            latent_dim=64,
            hidden_dims=[128],
            dropout=0.3,
            beta=0.2,
        )

        trainer = VAETrainer(model, device, lr=0.001)

        dataset = UserInteractionDataset(train_matrix)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        # Train for 2 epochs
        for _ in range(2):
            metrics = trainer.train_epoch(loader)
            assert metrics["total_loss"] > 0

        # Step 4: Generate recommendations
        model.eval()
        with torch.no_grad():
            # Get recommendations for first user
            user_idx = 0
            user_vector = torch.FloatTensor(train_matrix[user_idx].toarray().flatten())
            user_vector = user_vector.unsqueeze(0).to(device)

            # Get scores
            user_embedding = model.get_user_embedding(user_vector)
            scores = model.decode(user_embedding)
            scores = scores.squeeze().cpu().numpy()

            # Exclude seen items
            seen_items = train_matrix[user_idx].nonzero()[1]
            scores[seen_items] = -np.inf

            # Get top-5 recommendations
            top_k = 5
            top_indices = np.argsort(scores)[::-1][:top_k]

            assert len(top_indices) == top_k
            assert all(idx not in seen_items for idx in top_indices)

        # Step 5: Test model save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pth"

            # Save
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "n_items": n_items,
                    "latent_dim": 64,
                    "hidden_dims": [128],
                    "dropout": 0.3,
                    "beta": 0.2,
                },
            }
            torch.save(checkpoint, model_path)

            # Load
            loaded_checkpoint = torch.load(model_path, map_location=device)
            loaded_model = HybridVAE(
                n_items=loaded_checkpoint["config"]["n_items"],
                item_embeddings=embeddings,
                latent_dim=loaded_checkpoint["config"]["latent_dim"],
                hidden_dims=loaded_checkpoint["config"]["hidden_dims"],
                dropout=loaded_checkpoint["config"]["dropout"],
                beta=loaded_checkpoint["config"]["beta"],
            )
            loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])

            # Verify loaded model produces same output
            loaded_model.to(device)
            loaded_model.eval()
            with torch.no_grad():
                loaded_scores = loaded_model.decode(loaded_model.get_user_embedding(user_vector))
                loaded_scores = loaded_scores.squeeze().cpu().numpy()

                # Apply same masking as original scores
                loaded_scores[seen_items] = -np.inf

            np.testing.assert_array_almost_equal(scores, loaded_scores, decimal=5)


class TestMockEmbedder:
    """Test with a mock embedding generator."""

    def test_mock_sbert_embedder(self):
        """Test that we can mock the SBERT embedder for fast testing."""

        class MockEmbeddingGenerator:
            """Mock SBERT embedder that returns random normalized vectors."""

            def __init__(self, embedding_dim: int = 384):
                self.embedding_dim = embedding_dim

            def encode_texts(
                self, texts: list[str], batch_size: int = 32, show_progress: bool = False
            ) -> np.ndarray:
                """Return random normalized embeddings."""
                np.random.seed(hash(tuple(texts)) % 2**32)
                embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
                return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            def create_item_embeddings(
                self, df: pd.DataFrame, text_column: str = "item_text", item_id_column: str = "asin"
            ) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
                """Create embeddings for unique items."""
                unique_items = df[item_id_column].unique()
                texts = [f"Text for {item}" for item in unique_items]

                embeddings = self.encode_texts(texts)
                item_to_idx = {item: i for i, item in enumerate(unique_items)}
                idx_to_item = {i: item for item, i in item_to_idx.items()}

                return embeddings, item_to_idx, idx_to_item

        # Test mock embedder
        embedder = MockEmbeddingGenerator(embedding_dim=384)

        # Create test data
        df = pd.DataFrame(
            {
                "asin": ["item_1", "item_2", "item_3", "item_1", "item_2"],
                "item_text": ["text1", "text2", "text3", "text1", "text2"],
            }
        )

        embeddings, item_to_idx, idx_to_item = embedder.create_item_embeddings(df)

        assert embeddings.shape == (3, 384)  # 3 unique items
        assert len(item_to_idx) == 3
        assert len(idx_to_item) == 3

        # Verify normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
