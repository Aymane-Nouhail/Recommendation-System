"""
End-to-end integration test for the full recommendation pipeline.

Tests the complete flow from raw JSONL data through training and evaluation,
using synthetic data generated in memory. No external files required.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.ml.model import HybridVAE
from src.ml.train import UserInteractionDataset, VAETrainer

# Import pipeline components
from src.preprocessing.dataset import DatasetBuilder


class MockSentenceTransformer:
    """Mock SBERT model that returns deterministic embeddings without downloading."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedding_dim = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in sentences:
            # Deterministic seed from text
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.embedding_dim).astype(np.float32)
            if normalize_embeddings:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)


def generate_synthetic_jsonl_data(
    n_users: int = 30, n_items: int = 60, n_interactions: int = 400
) -> list[dict]:
    """
    Generate synthetic Amazon-style review data.

    Creates realistic-looking data with:
    - User IDs, Item ASINs
    - Ratings, timestamps
    - Titles, review text
    """
    np.random.seed(42)

    users = [f"USER_{i:04d}" for i in range(n_users)]
    items = [f"B{i:09d}" for i in range(n_items)]  # ASIN format

    titles = [
        "Amazing Product",
        "Great Quality",
        "Good Value",
        "Nice Item",
        "Excellent Purchase",
        "Worth the Money",
        "Highly Recommend",
        "Perfect Gift",
        "Best Ever",
        "Love It",
    ]

    review_templates = [
        "This product is {adj}. I {verb} it {adv}.",
        "Really {adj} quality. Would {verb} again.",
        "{adj} product, {verb} as expected.",
        "I'm {adv} {adj} with this purchase.",
    ]

    adjectives = ["great", "good", "excellent", "wonderful", "fantastic", "decent", "okay"]
    verbs = ["recommend", "buy", "use", "love", "enjoy"]
    adverbs = ["really", "very", "quite", "absolutely", "totally"]

    records = []
    seen_pairs = set()

    while len(records) < n_interactions:
        user = np.random.choice(users)
        item = np.random.choice(items)

        # Skip duplicate user-item pairs
        if (user, item) in seen_pairs:
            continue
        seen_pairs.add((user, item))

        # Generate rating (skewed towards positive)
        rating = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.05, 0.1, 0.15, 0.3, 0.4])

        # Generate timestamp (last 2 years)
        timestamp = np.random.randint(1640000000, 1700000000)

        # Generate title and text
        title = f"{np.random.choice(titles)} - Item {item[-4:]}"
        template = np.random.choice(review_templates)
        text = template.format(
            adj=np.random.choice(adjectives),
            verb=np.random.choice(verbs),
            adv=np.random.choice(adverbs),
        )

        records.append(
            {
                "user_id": user,
                "asin": item,
                "rating": rating,
                "timestamp": timestamp,
                "title": title,
                "text": text,
                "parent_asin": item,  # Same as asin for simplicity
            }
        )

    return records


class TestEndToEndPipeline:
    """End-to-end test of the full pipeline with synthetic data."""

    @pytest.fixture
    def synthetic_data(self) -> list[dict]:
        """Generate synthetic JSONL-style data."""
        return generate_synthetic_jsonl_data(n_users=30, n_items=60, n_interactions=400)

    @pytest.fixture
    def temp_workspace(self, tmp_path: Path) -> dict[str, Path]:
        """Create temporary directory structure mimicking the project."""
        data_dir = tmp_path / "data"
        embeddings_dir = tmp_path / "embeddings"
        models_dir = tmp_path / "models"

        data_dir.mkdir()
        embeddings_dir.mkdir()
        models_dir.mkdir()
        (models_dir / "figures").mkdir()

        return {
            "root": tmp_path,
            "data": data_dir,
            "embeddings": embeddings_dir,
            "models": models_dir,
        }

    @pytest.fixture
    def device(self) -> torch.device:
        """Get best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_full_pipeline_in_memory(self, synthetic_data, temp_workspace, device):
        """
        Test complete pipeline: raw data → cleaning → dataset → embeddings → training → evaluation.

        All operations happen in memory or temp directories. No real files needed.
        """
        data_dir = temp_workspace["data"]
        embeddings_dir = temp_workspace["embeddings"]
        models_dir = temp_workspace["models"]

        # =====================================================================
        # STEP 1: Write synthetic data to JSONL (simulating raw input)
        # =====================================================================
        raw_file = data_dir / "raw_reviews.jsonl"
        with open(raw_file, "w") as f:
            for record in synthetic_data:
                f.write(json.dumps(record) + "\n")

        assert raw_file.exists()
        print(f"\n✓ Step 1: Created synthetic JSONL with {len(synthetic_data)} records")

        # =====================================================================
        # STEP 2: Clean data (like cleaning.py)
        # =====================================================================
        df = pd.DataFrame(synthetic_data)

        # Add required fields
        df["binary_rating"] = (df["rating"] >= 3.0).astype(float)
        df["item_text"] = df.apply(
            lambda r: f"{r['title']}. {r['text']}" if pd.notna(r["title"]) else r["text"], axis=1
        )

        # Filter by min interactions
        original_len = len(df)
        for _ in range(10):  # Iterative filtering
            user_counts = df["user_id"].value_counts()
            item_counts = df["asin"].value_counts()
            valid_users = user_counts[user_counts >= 3].index
            valid_items = item_counts[item_counts >= 3].index
            df_filtered = df[df["user_id"].isin(valid_users) & df["asin"].isin(valid_items)]
            if len(df_filtered) == len(df):
                break
            df = df_filtered

        cleaned_file = data_dir / "cleaned_reviews.csv"
        df.to_csv(cleaned_file, index=False)

        assert len(df) > 0, "Filtering removed all data"
        print(f"✓ Step 2: Cleaned data: {original_len} → {len(df)} interactions")

        # =====================================================================
        # STEP 3: Build dataset splits (like dataset.py)
        # =====================================================================
        builder = DatasetBuilder()
        train_df, val_df, test_df = builder.leave_one_out_split(df)
        builder.fit_encoders(train_df)
        train_matrix = builder.create_interaction_matrix(train_df)

        n_users = len(builder.user_to_idx)
        n_items = len(builder.item_to_idx)

        # Save dataset artifacts
        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        with open(data_dir / "interaction_matrix.pkl", "wb") as f:
            pickle.dump(train_matrix, f)

        mappings = {
            "user_to_idx": builder.user_to_idx,
            "item_to_idx": builder.item_to_idx,
            "idx_to_user": builder.idx_to_user,
            "idx_to_item": builder.idx_to_item,
        }
        with open(data_dir / "mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)

        assert (data_dir / "train.csv").exists()
        assert (data_dir / "interaction_matrix.pkl").exists()
        print(
            f"✓ Step 3: Built dataset: {n_users} users, {n_items} items, "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        # =====================================================================
        # STEP 4: Generate embeddings (mocking SBERT)
        # =====================================================================
        mock_sbert = MockSentenceTransformer()

        # Get unique items and their text
        item_texts = train_df.groupby("asin")["item_text"].first()
        items_ordered = list(builder.item_to_idx.keys())
        texts = [item_texts.get(item, "[NO TEXT]") for item in items_ordered]

        embeddings = mock_sbert.encode(texts, normalize_embeddings=True)

        embeddings_file = embeddings_dir / "item_embeddings.npy"
        np.save(embeddings_file, embeddings)

        # Save item mapping
        with open(embeddings_dir / "item_to_idx.pkl", "wb") as f:
            pickle.dump(builder.item_to_idx, f)

        assert embeddings.shape == (n_items, 384)
        print(f"✓ Step 4: Generated embeddings: {embeddings.shape}")

        # =====================================================================
        # STEP 5: Train model (like train.py)
        # =====================================================================
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # Train for a few epochs
        n_epochs = 3
        for _epoch in range(n_epochs):
            metrics = trainer.train_epoch(loader)

        # Save model checkpoint
        model_path = models_dir / "best_model.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "n_items": n_items,
                "latent_dim": 64,
                "hidden_dims": [128],
                "dropout": 0.3,
                "beta": 0.2,
            },
            "metrics": metrics,
        }
        torch.save(checkpoint, model_path)

        assert model_path.exists()
        print(
            f"✓ Step 5: Trained model for {n_epochs} epochs, " f"loss={metrics['total_loss']:.4f}"
        )

        # =====================================================================
        # STEP 6: Evaluate model (like evaluate.py)
        # =====================================================================
        model.eval()

        # Simple evaluation: generate recommendations for test users
        hits_at_10 = 0
        n_test_users = 0

        test_user_items = test_df.groupby("user_id")["asin"].apply(set).to_dict()

        with torch.no_grad():
            for user_id, test_items in test_user_items.items():
                if user_id not in builder.user_to_idx:
                    continue

                user_idx = builder.user_to_idx[user_id]
                user_vector = (
                    torch.FloatTensor(train_matrix[user_idx].toarray().flatten())
                    .unsqueeze(0)
                    .to(device)
                )

                # Get scores
                z = model.get_user_embedding(user_vector)
                scores = model.decode(z).squeeze().cpu().numpy()

                # Exclude training items
                train_items = train_matrix[user_idx].nonzero()[1]
                scores[train_items] = -np.inf

                # Get top-10 recommendations
                top_10 = set(np.argsort(scores)[::-1][:10])

                # Check if any test item is in top-10
                test_item_indices = {
                    builder.item_to_idx[it] for it in test_items if it in builder.item_to_idx
                }
                if top_10 & test_item_indices:
                    hits_at_10 += 1
                n_test_users += 1

        hit_rate = hits_at_10 / n_test_users if n_test_users > 0 else 0

        # Save evaluation results
        eval_results = {
            "hit_rate_at_10": hit_rate,
            "n_test_users": n_test_users,
            "hits": hits_at_10,
        }
        with open(models_dir / "figures" / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"✓ Step 6: Evaluated model, HR@10={hit_rate:.4f} ({hits_at_10}/{n_test_users})")

        # =====================================================================
        # STEP 7: Load model and verify inference (like API would)
        # =====================================================================
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
        loaded_model.to(device)
        loaded_model.eval()

        # Generate a recommendation
        with torch.no_grad():
            test_user_idx = 0
            user_vector = (
                torch.FloatTensor(train_matrix[test_user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )

            scores = loaded_model.decode(loaded_model.get_user_embedding(user_vector))
            top_5 = np.argsort(scores.squeeze().cpu().numpy())[::-1][:5]

            recommended_items = [builder.idx_to_item[idx] for idx in top_5]

        assert len(recommended_items) == 5
        print(f"✓ Step 7: Loaded model and generated recommendations: {recommended_items}")

        # =====================================================================
        # VERIFY: All expected output files exist
        # =====================================================================
        expected_files = [
            data_dir / "cleaned_reviews.csv",
            data_dir / "train.csv",
            data_dir / "val.csv",
            data_dir / "test.csv",
            data_dir / "interaction_matrix.pkl",
            data_dir / "mappings.pkl",
            embeddings_dir / "item_embeddings.npy",
            models_dir / "best_model.pth",
            models_dir / "figures" / "evaluation_results.json",
        ]

        for f in expected_files:
            assert f.exists(), f"Missing: {f}"

        print("\n✓ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
        print(f"  - Data: {n_users} users, {n_items} items")
        print("  - Model: 64-dim latent, [128] hidden")
        print(f"  - Evaluation: HR@10 = {hit_rate:.4f}")


class TestPipelineComponents:
    """Test individual pipeline components work correctly."""

    def test_synthetic_data_generation(self):
        """Test that synthetic data has correct format."""
        data = generate_synthetic_jsonl_data(n_users=10, n_items=20, n_interactions=50)

        assert len(data) == 50
        assert all("user_id" in r for r in data)
        assert all("asin" in r for r in data)
        assert all("rating" in r for r in data)
        assert all("timestamp" in r for r in data)
        assert all("title" in r for r in data)
        assert all("text" in r for r in data)

    def test_mock_sbert_deterministic(self):
        """Test that mock SBERT produces consistent embeddings."""
        mock = MockSentenceTransformer()

        text = "This is a test sentence"
        emb1 = mock.encode([text])
        emb2 = mock.encode([text])

        np.testing.assert_array_equal(emb1, emb2)

    def test_mock_sbert_normalized(self):
        """Test that mock SBERT embeddings are normalized."""
        mock = MockSentenceTransformer()

        texts = ["Test 1", "Test 2", "Test 3"]
        embeddings = mock.encode(texts, normalize_embeddings=True)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
