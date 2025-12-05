"""
Visualization module for recommendation system.

Generates plots from saved training history and baseline results.
Can be run standalone via `make visualize` after training.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Training Visualizations
# =============================================================================


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_recon_losses: List[float],
    train_kl_losses: List[float],
    output_path: Path,
) -> None:
    """
    Generate training visualization plots for the report.

    Args:
        train_losses: List of training total losses per epoch
        val_losses: List of validation total losses per epoch
        train_recon_losses: List of reconstruction losses per epoch
        train_kl_losses: List of KL divergence losses per epoch
        output_path: Directory to save plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = range(1, len(train_losses) + 1)

    # Plot 1: Training vs Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = val_losses.index(min(val_losses)) + 1
    ax.axvline(
        x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (epoch {best_epoch})"
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved loss curves to {output_path / 'loss_curves.png'}")

    # Plot 2: Loss Components Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_recon_losses, "b-", label="Reconstruction Loss", linewidth=2)
    ax.plot(epochs, train_kl_losses, "orange", label="KL Divergence", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss Component", fontsize=12)
    ax.set_title("Loss Components: Reconstruction vs KL Divergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "loss_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved loss components to {output_path / 'loss_components.png'}")

    # Plot 3: Combined 2x2 subplot figure for report
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Train vs Val loss
    axes[0, 0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_losses, "r-", label="Val", linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Reconstruction loss
    axes[0, 1].plot(epochs, train_recon_losses, "b-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reconstruction Loss")
    axes[0, 1].set_title("Reconstruction Loss (Multinomial NLL)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: KL Divergence
    axes[1, 0].plot(epochs, train_kl_losses, "orange", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("KL Divergence (Regularization)")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Loss ratio
    recon_ratio = [
        r / (r + k) if (r + k) > 0 else 0.5 for r, k in zip(train_recon_losses, train_kl_losses)
    ]
    axes[1, 1].plot(epochs, recon_ratio, "purple", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Reconstruction / Total")
    axes[1, 1].set_title("Reconstruction Loss Ratio")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("HybridVAE Training Summary", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path / "training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training summary to {output_path / 'training_summary.png'}")


def plot_latent_space(
    model_path: Path,
    data_dir: Path,
    embeddings_path: Path,
    output_path: Path,
    n_samples: int = 2000,
) -> None:
    """
    Visualize the learned latent space using t-SNE and PCA.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing processed dataset
        embeddings_path: Path to item embeddings
        output_path: Directory to save plots
        n_samples: Maximum number of users to visualize
    """
    from src.ml.model import HybridVAE
    from src.preprocessing.embeddings import load_embeddings

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})

    # Load embeddings
    embeddings_path_obj = Path(embeddings_path)
    mappings_path = embeddings_path_obj.with_name(embeddings_path_obj.stem + "_mappings.pkl")
    embeddings, _, _ = load_embeddings(str(embeddings_path), str(mappings_path))

    # Create model
    n_items = model_config.get("n_items", embeddings.shape[0])
    latent_dim = model_config.get("latent_dim", 64)
    hidden_dims = model_config.get("hidden_dims", [256])

    model = HybridVAE(
        n_items=n_items,
        item_embeddings=embeddings,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load interaction matrix
    with open(data_dir / "interaction_matrix.pkl", "rb") as f:
        interaction_matrix = pickle.load(f)

    # Sample users
    n_users = interaction_matrix.shape[0]
    if n_users > n_samples:
        sampled_indices = np.random.choice(n_users, n_samples, replace=False)
    else:
        sampled_indices = np.arange(n_users)

    # Get user embeddings
    embeddings_list = []
    activity_levels = []

    with torch.no_grad():
        for user_idx in tqdm(sampled_indices, desc="Computing user embeddings"):
            user_vector = (
                torch.FloatTensor(interaction_matrix[user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )
            mu, _ = model.encode(user_vector)
            embeddings_list.append(mu.cpu().numpy().flatten())
            activity_levels.append(interaction_matrix[user_idx].nnz)

    user_embeddings = np.array(embeddings_list)
    activity_levels = np.array(activity_levels)
    log_activity = np.log1p(activity_levels)

    # PCA visualization
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(user_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Log(1 + #Interactions)", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    ax.set_title("User Latent Space (PCA)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path / "latent_space_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PCA plot to {output_path / 'latent_space_pca.png'}")

    # t-SNE visualization
    logger.info("Computing t-SNE projection (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(user_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Log(1 + #Interactions)", fontsize=11)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("User Latent Space (t-SNE)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path / "latent_space_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved t-SNE plot to {output_path / 'latent_space_tsne.png'}")

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = axes[0].scatter(
        pca_result[:, 0], pca_result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    axes[0].set_title("PCA Projection")
    plt.colorbar(scatter1, ax=axes[0], label="Log(1 + #Interactions)")

    scatter2 = axes[1].scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
    )
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].set_title("t-SNE Projection")
    plt.colorbar(scatter2, ax=axes[1], label="Log(1 + #Interactions)")

    plt.suptitle("User Latent Space Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path / "latent_space_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined latent space plot to {output_path / 'latent_space_combined.png'}")


# =============================================================================
# Baseline Visualizations
# =============================================================================


def plot_baseline_comparison(
    all_results: Dict[str, Dict],
    k_values: List[int],
    output_path: Path,
    n_negatives: Optional[int] = 99,
) -> None:
    """
    Generate comparison visualizations for baseline models.

    Args:
        all_results: Dictionary mapping model names to their metrics
        k_values: List of K values used in evaluation
        output_path: Directory to save plots
        n_negatives: Number of negatives used (for plot title)
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    model_names = list(all_results.keys())
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]

    protocol = f"Negative Sampling ({n_negatives})" if n_negatives else "Full Ranking"

    # Plot 1: Recall@K comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.25
    multiplier = 0

    for i, model_name in enumerate(model_names):
        recalls = [all_results[model_name][k]["recall"] for k in k_values]
        offset = width * multiplier
        ax.bar(x + offset, recalls, width, label=model_name, color=colors[i % len(colors)])
        multiplier += 1

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title(f"Recall@K Comparison ({protocol})", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path / "baseline_recall_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved recall comparison to {output_path / 'baseline_recall_comparison.png'}")

    # Plot 2: NDCG@K comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    multiplier = 0

    for i, model_name in enumerate(model_names):
        ndcgs = [all_results[model_name][k]["ndcg"] for k in k_values]
        offset = width * multiplier
        ax.bar(x + offset, ndcgs, width, label=model_name, color=colors[i % len(colors)])
        multiplier += 1

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("NDCG@K", fontsize=12)
    ax.set_title(f"NDCG@K Comparison ({protocol})", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path / "baseline_ndcg_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved NDCG comparison to {output_path / 'baseline_ndcg_comparison.png'}")

    # Plot 3: Hit Ratio@K comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    multiplier = 0

    for i, model_name in enumerate(model_names):
        hit_ratios = [all_results[model_name][k]["hit_ratio"] for k in k_values]
        offset = width * multiplier
        ax.bar(x + offset, hit_ratios, width, label=model_name, color=colors[i % len(colors)])
        multiplier += 1

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Hit Ratio@K", fontsize=12)
    ax.set_title(f"Hit Ratio@K Comparison ({protocol})", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path / "baseline_hitrate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved hit ratio comparison to {output_path / 'baseline_hitrate_comparison.png'}")

    # Plot 4: Combined summary figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Recall@10 bar chart
    k_main = 10
    recalls = [all_results[m][k_main]["recall"] for m in model_names]
    bars = axes[0, 0].bar(model_names, recalls, color=colors[: len(model_names)])
    axes[0, 0].set_ylabel("Recall@10")
    axes[0, 0].set_title("Recall@10 by Model")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, recalls):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Top-right: NDCG@10 bar chart
    ndcgs = [all_results[m][k_main]["ndcg"] for m in model_names]
    bars = axes[0, 1].bar(model_names, ndcgs, color=colors[: len(model_names)])
    axes[0, 1].set_ylabel("NDCG@10")
    axes[0, 1].set_title("NDCG@10 by Model")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, ndcgs):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Bottom-left: Hit Ratio@10 bar chart
    hit_ratios = [all_results[m][k_main]["hit_ratio"] for m in model_names]
    bars = axes[1, 0].bar(model_names, hit_ratios, color=colors[: len(model_names)])
    axes[1, 0].set_ylabel("Hit Ratio@10")
    axes[1, 0].set_title("Hit Ratio@10 by Model")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, hit_ratios):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Bottom-right: Recall across K values (line plot)
    for i, model_name in enumerate(model_names):
        recalls_line = [all_results[model_name][k]["recall"] for k in k_values]
        axes[1, 1].plot(
            k_values,
            recalls_line,
            "o-",
            label=model_name,
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
        )
    axes[1, 1].set_xlabel("K")
    axes[1, 1].set_ylabel("Recall@K")
    axes[1, 1].set_title("Recall@K Trend")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Baseline Models Comparison ({protocol})", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path / "baseline_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved baseline summary to {output_path / 'baseline_summary.png'}")


# =============================================================================
# CLI Entry Points
# =============================================================================


def visualize_training(model_dir: str, data_dir: str, embeddings_path: str) -> None:
    """Generate all training visualizations from saved history."""
    model_path = Path(model_dir)
    figures_path = model_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # Load training history
    history_path = model_path / "training_history.json"
    if not history_path.exists():
        logger.error(f"Training history not found: {history_path}")
        logger.error("Run `make train` first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    logger.info("Generating training curve visualizations...")
    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        history["train_recon_losses"],
        history["train_kl_losses"],
        figures_path,
    )

    # Generate latent space visualization if model exists
    best_model_path = model_path / "best_model.pth"
    if best_model_path.exists():
        logger.info("Generating latent space visualizations...")
        plot_latent_space(
            best_model_path,
            Path(data_dir),
            Path(embeddings_path),
            figures_path,
            n_samples=2000,
        )
    else:
        logger.warning(f"Model not found: {best_model_path}, skipping latent space plots")

    logger.info(f"All visualizations saved to {figures_path}")


def visualize_baselines(model_dir: str) -> None:
    """Generate baseline comparison visualizations from saved results."""
    figures_path = Path(model_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # Load baseline results
    results_path = figures_path / "baseline_results.json"
    if not results_path.exists():
        logger.error(f"Baseline results not found: {results_path}")
        logger.error("Run `make baseline` first.")
        return

    with open(results_path) as f:
        results_raw = json.load(f)

    # Convert string keys back to int
    all_results = {}
    for model_name, metrics in results_raw.items():
        all_results[model_name] = {int(k): v for k, v in metrics.items()}

    k_values = sorted(list(next(iter(all_results.values())).keys()))

    logger.info("Generating baseline comparison visualizations...")
    plot_baseline_comparison(all_results, k_values, figures_path)
    logger.info(f"All visualizations saved to {figures_path}")


# =============================================================================
# Grid Search Visualizations
# =============================================================================


def plot_grid_search_results(results_path: Path, output_path: Path) -> None:
    """
    Generate visualizations for hyperparameter grid search results.

    Creates:
    1. Parameter importance bar chart
    2. Top configs comparison
    3. Hyperparameter heatmaps

    Args:
        results_path: Path to grid_search_results.json
        output_path: Directory to save plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    with open(results_path) as f:
        data = json.load(f)

    all_results = [r for r in data["all_results"] if "error" not in r]
    if not all_results:
        logger.error("No valid results found in grid search")
        return

    # Extract data for plotting
    configs = [r["config"] for r in all_results]
    ndcg_scores = [r["ndcg@10"] for r in all_results]
    recall_scores = [r["recall@10"] for r in all_results]

    # Plot 1: Top 10 configurations
    sorted_results = sorted(all_results, key=lambda x: x["ndcg@10"], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"Config {i+1}" for i in range(len(sorted_results))]
    ndcg_vals = [r["ndcg@10"] for r in sorted_results]
    recall_vals = [r["recall@10"] for r in sorted_results]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ndcg_vals, width, label="NDCG@10", color="#3498db")
    ax.bar(x + width / 2, recall_vals, width, label="Recall@10", color="#2ecc71")

    ax.set_xlabel("Configuration Rank", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Top 10 Hyperparameter Configurations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path / "grid_search_top_configs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved top configs plot to {output_path / 'grid_search_top_configs.png'}")

    # Plot 2: Parameter impact analysis (box plots)
    param_names = list(configs[0].keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(param_names):
        if idx >= len(axes):
            break

        # Group results by parameter value
        param_values = {}
        for cfg, ndcg in zip(configs, ndcg_scores):
            val = str(cfg[param])
            if val not in param_values:
                param_values[val] = []
            param_values[val].append(ndcg)

        # Create box plot
        labels = list(param_values.keys())
        box_data = [param_values[label] for label in labels]

        bp = axes[idx].boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)

        axes[idx].set_title(f"{param}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("NDCG@10")
        axes[idx].grid(True, alpha=0.3, axis="y")
        axes[idx].tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for idx in range(len(param_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Hyperparameter Impact on NDCG@10", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path / "grid_search_param_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved parameter impact plot to {output_path / 'grid_search_param_impact.png'}")

    # Plot 3: Latent dim vs Beta heatmap (if both exist)
    if "latent_dim" in param_names and "beta" in param_names:
        latent_dims = sorted(set(c["latent_dim"] for c in configs))
        betas = sorted(set(c["beta"] for c in configs))

        # Average NDCG for each (latent_dim, beta) combination
        heatmap_data = np.zeros((len(latent_dims), len(betas)))
        counts = np.zeros((len(latent_dims), len(betas)))

        for config, ndcg in zip(configs, ndcg_scores):
            i = latent_dims.index(config["latent_dim"])
            j = betas.index(config["beta"])
            heatmap_data[i, j] += ndcg
            counts[i, j] += 1

        heatmap_data = heatmap_data / np.maximum(counts, 1)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")

        ax.set_xticks(np.arange(len(betas)))
        ax.set_yticks(np.arange(len(latent_dims)))
        ax.set_xticklabels([f"{b}" for b in betas])
        ax.set_yticklabels([f"{ld}" for ld in latent_dims])
        ax.set_xlabel("Beta (KL Weight)", fontsize=12)
        ax.set_ylabel("Latent Dimension", fontsize=12)
        ax.set_title("Average NDCG@10: Latent Dim vs Beta", fontsize=14, fontweight="bold")

        # Add value annotations
        for i in range(len(latent_dims)):
            for j in range(len(betas)):
                ax.text(
                    j,
                    i,
                    f"{heatmap_data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        plt.colorbar(im, ax=ax, label="NDCG@10")
        plt.tight_layout()
        fig.savefig(output_path / "grid_search_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved heatmap to {output_path / 'grid_search_heatmap.png'}")

    # Plot 4: Summary with best config annotation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(recall_scores, ndcg_scores, alpha=0.6, s=50, c="#3498db")

    # Highlight best
    best_idx = ndcg_scores.index(max(ndcg_scores))
    ax.scatter(
        [recall_scores[best_idx]],
        [ndcg_scores[best_idx]],
        s=200,
        c="#e74c3c",
        marker="*",
        label="Best Config",
        zorder=5,
    )

    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title("Grid Search Results: Recall vs NDCG", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate best config
    best_config = data["best_config"]
    annotation = "\n".join([f"{k}: {v}" for k, v in best_config.items()])
    ax.annotate(
        f"Best:\n{annotation}",
        xy=(recall_scores[best_idx], ndcg_scores[best_idx]),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="yellow", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "grid_search_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter plot to {output_path / 'grid_search_scatter.png'}")


def visualize_grid_search(model_dir: str) -> None:
    """Generate grid search visualizations from saved results."""
    figures_path = Path(model_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(model_dir) / "grid_search_results.json"
    if not results_path.exists():
        logger.error(f"Grid search results not found: {results_path}")
        logger.error("Run `make tune` first.")
        return

    logger.info("Generating grid search visualizations...")
    plot_grid_search_results(results_path, figures_path)
    logger.info(f"All visualizations saved to {figures_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for recommendation system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Visualization type")

    # Training visualizations
    train_parser = subparsers.add_parser("training", help="Generate training visualizations")
    train_parser.add_argument("--model-dir", default=str(config.MODEL_DIR), help="Model directory")
    train_parser.add_argument("--data-dir", default=str(config.DATA_DIR), help="Data directory")
    train_parser.add_argument(
        "--embeddings", default=str(config.EMBEDDINGS_FILE), help="Embeddings path"
    )

    # Baseline visualizations
    baseline_parser = subparsers.add_parser("baseline", help="Generate baseline visualizations")
    baseline_parser.add_argument(
        "--model-dir", default=str(config.MODEL_DIR), help="Model directory"
    )

    # Tuning visualizations
    tune_parser = subparsers.add_parser("tuning", help="Generate grid search visualizations")
    tune_parser.add_argument("--model-dir", default=str(config.MODEL_DIR), help="Model directory")

    # All visualizations
    all_parser = subparsers.add_parser("all", help="Generate all visualizations")
    all_parser.add_argument("--model-dir", default=str(config.MODEL_DIR), help="Model directory")
    all_parser.add_argument("--data-dir", default=str(config.DATA_DIR), help="Data directory")
    all_parser.add_argument(
        "--embeddings", default=str(config.EMBEDDINGS_FILE), help="Embeddings path"
    )

    args = parser.parse_args()

    if args.command == "training":
        visualize_training(args.model_dir, args.data_dir, args.embeddings)
    elif args.command == "baseline":
        visualize_baselines(args.model_dir)
    elif args.command == "tuning":
        visualize_grid_search(args.model_dir)
    elif args.command == "all":
        visualize_training(args.model_dir, args.data_dir, args.embeddings)
        visualize_baselines(args.model_dir)
        visualize_grid_search(args.model_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
