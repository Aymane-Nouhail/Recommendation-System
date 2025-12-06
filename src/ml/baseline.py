"""
Baseline models for recommendation system comparison.

Implements: Popularity, Item-KNN, SVD, Mult-VAE, and LightGCN.
"""

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml.evaluate import hit_ratio_at_k, ndcg_at_k, recall_at_k
from ml.train import load_training_data
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Base Class
# =============================================================================


class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""

    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix) -> None:
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, user_idx: int) -> np.ndarray:
        """Predict scores for all items for a user."""
        pass


# =============================================================================
# Simple Baselines
# =============================================================================


class PopularityRecommender(BaseRecommender):
    """Non-personalized recommender ranking items by popularity."""

    def __init__(self):
        self.scores = None

    def fit(self, interaction_matrix: csr_matrix) -> None:
        logger.info("Fitting Popularity baseline...")
        self.scores = np.array(interaction_matrix.sum(axis=0)).flatten()

    def predict(self, user_idx: int) -> np.ndarray:
        return self.scores.copy()


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with cosine similarity."""

    def __init__(self, k: int = 50):
        self.k = k
        self.similarity = None
        self.matrix = None

    def fit(self, interaction_matrix: csr_matrix) -> None:
        logger.info(f"Fitting Item-KNN (k={self.k})...")
        self.matrix = interaction_matrix
        n_items = interaction_matrix.shape[1]

        if n_items <= 10000:
            item_matrix = interaction_matrix.T.toarray()
            self.similarity = cosine_similarity(item_matrix)
            np.fill_diagonal(self.similarity, 0)

    def predict(self, user_idx: int) -> np.ndarray:
        interacted = self.matrix[user_idx].nonzero()[1]
        if len(interacted) == 0:
            return np.zeros(self.matrix.shape[1])

        if self.similarity is not None:
            return self.similarity[:, interacted].sum(axis=1)

        # On-the-fly for large matrices
        item_matrix = self.matrix.T
        return cosine_similarity(item_matrix, item_matrix[interacted]).sum(axis=1)


class SVDRecommender(BaseRecommender):
    """Matrix factorization using TruncatedSVD."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None

    def fit(self, interaction_matrix: csr_matrix) -> None:
        logger.info(f"Fitting SVD ({self.n_components} components)...")
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_
        logger.info(f"Explained variance: {self.model.explained_variance_ratio_.sum():.4f}")

    def predict(self, user_idx: int) -> np.ndarray:
        return self.user_factors[user_idx] @ self.item_factors


# =============================================================================
# Mult-VAE Baseline
# =============================================================================


class MultVAE(nn.Module):
    """Multinomial Variational Autoencoder for collaborative filtering."""

    def __init__(
        self, n_items: int, hidden_dim: int = 600, latent_dim: int = 200, dropout: float = 0.5
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
        )
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.dropout(F.normalize(x, p=2, dim=1)))
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class MultVAERecommender(BaseRecommender):
    """Mult-VAE recommender wrapper."""

    def __init__(
        self,
        hidden_dim: int = 600,
        latent_dim: int = 200,
        epochs: int = 50,
        lr: float = 1e-3,
        beta: float = 0.2,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.model: MultVAE | None = None
        self.device = self._get_device()
        self.matrix: csr_matrix | None = None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, interaction_matrix: csr_matrix) -> None:
        logger.info(f"Fitting Mult-VAE (latent={self.latent_dim}, epochs={self.epochs})...")
        self.matrix = interaction_matrix
        n_items = interaction_matrix.shape[1]

        self.model = MultVAE(n_items, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Create dataloader
        data = torch.FloatTensor(interaction_matrix.toarray())
        loader = DataLoader(TensorDataset(data), batch_size=512, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                recon, mu, logvar = self.model(batch)
                recon_loss = -torch.mean(torch.sum(F.log_softmax(recon, dim=1) * batch, dim=1))
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()

    def predict(self, user_idx: int) -> np.ndarray:
        assert self.matrix is not None and self.model is not None, "Model not fitted"
        with torch.no_grad():
            user_vec = torch.FloatTensor(self.matrix[user_idx].toarray()).to(self.device)
            scores, _, _ = self.model(user_vec)
            return scores.cpu().numpy().flatten()


# =============================================================================
# LightGCN Baseline
# =============================================================================


class LightGCNModel(nn.Module):
    """LightGCN: Simplifying and Powering Graph Convolution Network."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def get_ego_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self, adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_emb = self.get_ego_embeddings()
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        embeddings_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embeddings_list.append(all_emb)

        all_emb = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return all_emb[: self.n_users], all_emb[self.n_users :]


class LightGCNRecommender(BaseRecommender):
    """LightGCN recommender wrapper."""

    def __init__(
        self,
        embedding_dim: int = 64,
        n_layers: int = 3,
        epochs: int = 50,
        lr: float = 1e-3,
        reg: float = 1e-4,
    ):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.model: LightGCNModel | None = None
        # Force CPU for LightGCN - sparse ops not supported on MPS
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.user_final_emb: torch.Tensor | None = None
        self.item_final_emb: torch.Tensor | None = None

    def _build_adj_matrix(self, interaction_matrix: csr_matrix) -> torch.Tensor:
        """Build normalized adjacency matrix for LightGCN."""
        n_users, n_items = interaction_matrix.shape[0], interaction_matrix.shape[1]
        n_nodes = n_users + n_items

        # Build bipartite adjacency
        rows, cols = interaction_matrix.nonzero()
        rows_top = rows
        cols_top = cols + n_users
        rows_bottom = cols + n_users
        cols_bottom = rows

        all_rows = np.concatenate([rows_top, rows_bottom])
        all_cols = np.concatenate([cols_top, cols_bottom])

        # Compute degree normalization
        adj = csr_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(n_nodes, n_nodes))
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        # Normalize: D^(-0.5) * A * D^(-0.5)
        norm_vals = d_inv_sqrt[all_rows] * d_inv_sqrt[all_cols]

        indices = torch.LongTensor([all_rows, all_cols])
        values = torch.FloatTensor(norm_vals)
        return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).to(self.device)

    def fit(self, interaction_matrix: csr_matrix) -> None:
        logger.info(f"Fitting LightGCN (dim={self.embedding_dim}, layers={self.n_layers})...")
        n_users, n_items = interaction_matrix.shape[0], interaction_matrix.shape[1]

        self.model = LightGCNModel(n_users, n_items, self.embedding_dim, self.n_layers).to(
            self.device
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        adj_matrix = self._build_adj_matrix(interaction_matrix)

        # Get positive pairs for BPR loss
        pos_users, pos_items = interaction_matrix.nonzero()
        n_interactions = len(pos_users)

        self.model.train()
        for epoch in range(self.epochs):
            # Shuffle interactions
            perm = np.random.permutation(n_interactions)
            pos_users_shuffled = pos_users[perm]
            pos_items_shuffled = pos_items[perm]

            total_loss = 0
            batch_size = 1024

            for start in range(0, n_interactions, batch_size):
                end = min(start + batch_size, n_interactions)
                batch_users = torch.LongTensor(pos_users_shuffled[start:end]).to(self.device)
                batch_pos_items = torch.LongTensor(pos_items_shuffled[start:end]).to(self.device)
                batch_neg_items = torch.LongTensor(np.random.randint(0, n_items, end - start)).to(
                    self.device
                )

                optimizer.zero_grad()
                user_emb, item_emb = self.model(adj_matrix)

                u_emb = user_emb[batch_users]
                pos_emb = item_emb[batch_pos_items]
                neg_emb = item_emb[batch_neg_items]

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)

                bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                reg_loss = (
                    self.reg
                    * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))
                    / len(batch_users)
                )
                loss = bpr_loss + reg_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

        # Store final embeddings
        self.model.eval()
        with torch.no_grad():
            self.user_final_emb, self.item_final_emb = self.model(adj_matrix)

    def predict(self, user_idx: int) -> np.ndarray:
        with torch.no_grad():
            user_emb = self.user_final_emb[user_idx]
            scores = (user_emb @ self.item_final_emb.T).cpu().numpy()
            return scores


# =============================================================================
# Evaluation
# =============================================================================


def _build_input_matrix(
    train_df: pd.DataFrame, val_df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict, shape: tuple
) -> tuple[csr_matrix, pd.DataFrame]:
    """Build sparse interaction matrix from train/val data."""
    train_pos = (
        train_df[train_df["binary_rating"] == 1]
        if "binary_rating" in train_df.columns
        else train_df
    )
    val_pos = val_df[val_df["binary_rating"] == 1] if "binary_rating" in val_df.columns else val_df
    combined = pd.concat([train_pos, val_pos])

    rows = combined["user_id"].map(user_to_idx)
    cols = combined["asin"].map(item_to_idx)
    matrix = csr_matrix((np.ones(len(combined)), (rows, cols)), shape=shape)
    return matrix, combined


def evaluate_model(
    model: BaseRecommender,
    name: str,
    matrix: csr_matrix,
    test_df: pd.DataFrame,
    user_to_idx: dict,
    item_to_idx: dict,
    k_values: list[int] | None = None,
    n_negatives: int | None = 99,
) -> dict[int, dict[str, float]]:
    """Evaluate a recommender model."""
    k_values = k_values or [5, 10, 20]
    protocol = f"negative sampling ({n_negatives})" if n_negatives else "full ranking"
    logger.info(f"Evaluating {name} with {protocol}...")

    results = {k: {"recall": [], "ndcg": [], "hit_ratio": []} for k in k_values}
    n_items = matrix.shape[1]
    all_items = np.arange(n_items)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {name}"):
        user_id, test_item_id = row["user_id"], row["asin"]
        if user_id not in user_to_idx or test_item_id not in item_to_idx:
            continue

        user_idx = user_to_idx[user_id]
        test_item_idx = item_to_idx[test_item_id]
        seen = set(matrix[user_idx].indices)
        scores = model.predict(user_idx)

        if n_negatives is not None:
            # Negative sampling
            mask = np.ones(n_items, dtype=bool)
            mask[list(seen)] = False
            mask[test_item_idx] = False
            available = all_items[mask]
            negatives = (
                available
                if len(available) < n_negatives
                else np.random.choice(available, n_negatives, replace=False)
            )
            candidates = np.concatenate([[test_item_idx], negatives])
            ranked = candidates[np.argsort(scores[candidates])[::-1]]
        else:
            # Full ranking
            scores[list(seen)] = -np.inf
            ranked = np.argsort(scores)[::-1][: max(k_values)]

        relevant = np.array([test_item_idx])
        for k in k_values:
            top_k = ranked[:k]
            results[k]["recall"].append(recall_at_k(top_k, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(top_k, relevant, k))
            results[k]["hit_ratio"].append(hit_ratio_at_k(top_k, relevant, k))

    return {
        k: {
            metric: float(np.mean(results[k][metric])) if results[k][metric] else 0.0
            for metric in ["recall", "ndcg", "hit_ratio"]
        }
        for k in k_values
    }


def run_all_baselines(
    data_dir: str, k_values: list[int] | None = None, n_negatives: int | None = 99
) -> dict:
    """Run all baseline models and compare."""
    k_values = k_values or [5, 10, 20]

    # Load data
    full_matrix, train_df, val_df, mappings = load_training_data(data_dir)
    user_to_idx, item_to_idx = mappings["user_to_idx"], mappings["item_to_idx"]
    input_matrix, _ = _build_input_matrix(
        train_df, val_df, user_to_idx, item_to_idx, full_matrix.shape
    )
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    # Define baselines
    baselines = {
        "Popularity": PopularityRecommender(),
        "Item-KNN": ItemKNNRecommender(k=50),
        "SVD": SVDRecommender(n_components=50),
        "Mult-VAE": MultVAERecommender(hidden_dim=600, latent_dim=200, epochs=50),
        "LightGCN": LightGCNRecommender(embedding_dim=64, n_layers=3, epochs=50),
    }

    all_results = {}
    for name, model in baselines.items():
        logger.info(f"\n{'='*60}\nTraining {name}\n{'='*60}")
        start_time = time.time()
        model.fit(input_matrix)
        training_time = round(time.time() - start_time, 2)
        logger.info(f"{name} training time: {training_time:.1f}s")
        results = evaluate_model(
            model, name, input_matrix, test_df, user_to_idx, item_to_idx, k_values, n_negatives
        )
        results["training_time_seconds"] = training_time
        all_results[name] = results

    # Print results
    protocol = f"negative sampling ({n_negatives})" if n_negatives else "full ranking"
    logger.info(f"\n{'='*70}\nBASELINE COMPARISON ({protocol})\n{'='*70}")
    print(
        f"\n{'-'*85}\n{'Model':<15} | {'Recall@5':>10} | {'Recall@10':>10} | {'Recall@20':>10} | {'NDCG@10':>10} | {'NDCG@20':>10}\n{'-'*85}"
    )
    for name, metrics in all_results.items():
        print(
            f"{name:<15} | {metrics[5]['recall']:>10.4f} | {metrics[10]['recall']:>10.4f} | {metrics[20]['recall']:>10.4f} | {metrics[10]['ndcg']:>10.4f} | {metrics[20]['ndcg']:>10.4f}"
        )
    print("-" * 85)

    # Save results
    figures_path = Path(data_dir).parent / "models" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    with open(figures_path / "baseline_results.json", "w") as f:
        json.dump(
            {m: {str(k): v for k, v in metrics.items()} for m, metrics in all_results.items()},
            f,
            indent=2,
        )
    logger.info(f"Saved to {figures_path / 'baseline_results.json'}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")
    parser.add_argument("--data", default=str(config.DATA_DIR))
    parser.add_argument("--n-negatives", type=int, default=99, help="0 for full ranking")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    n_negatives = args.n_negatives if args.n_negatives > 0 else None
    run_all_baselines(args.data, args.k_values, n_negatives)


if __name__ == "__main__":
    main()
