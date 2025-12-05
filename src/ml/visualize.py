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

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Consistent styling
COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
STYLE = "seaborn-v0_8-whitegrid"


def _save_figure(fig: Figure, path: Path, name: str) -> None:
    """Save figure and log."""
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {name} to {path}")


def _get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Training Visualizations
# =============================================================================


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_recon_losses: list[float],
    train_kl_losses: list[float],
    output_path: Path,
) -> None:
    """Generate training visualization plots."""
    plt.style.use(STYLE)
    epochs = range(1, len(train_losses) + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1

    # Plot 1: Training vs Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax.axvline(
        x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (epoch {best_epoch})"
    )
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training vs Validation Loss")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, output_path, "loss_curves.png")

    # Plot 2: Loss Components
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_recon_losses, "b-", label="Reconstruction Loss", linewidth=2)
    ax.plot(epochs, train_kl_losses, "orange", label="KL Divergence", linewidth=2)
    ax.set(
        xlabel="Epoch",
        ylabel="Loss Component",
        title="Loss Components: Reconstruction vs KL Divergence",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, output_path, "loss_components.png")

    # Plot 3: Combined 2x2 Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train vs Val
    axes[0, 0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_losses, "r-", label="Val", linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7)
    axes[0, 0].set(xlabel="Epoch", ylabel="Total Loss", title="Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction
    axes[0, 1].plot(epochs, train_recon_losses, "b-", linewidth=2)
    axes[0, 1].set(
        xlabel="Epoch", ylabel="Reconstruction Loss", title="Reconstruction Loss (Multinomial NLL)"
    )
    axes[0, 1].grid(True, alpha=0.3)

    # KL Divergence
    axes[1, 0].plot(epochs, train_kl_losses, "orange", linewidth=2)
    axes[1, 0].set(xlabel="Epoch", ylabel="KL Divergence", title="KL Divergence (Regularization)")
    axes[1, 0].grid(True, alpha=0.3)

    # Loss ratio
    recon_ratio = [
        r / (r + k) if (r + k) > 0 else 0.5 for r, k in zip(train_recon_losses, train_kl_losses)
    ]
    axes[1, 1].plot(epochs, recon_ratio, "purple", linewidth=2)
    axes[1, 1].set(
        xlabel="Epoch",
        ylabel="Reconstruction / Total",
        title="Reconstruction Loss Ratio",
        ylim=(0, 1),
    )
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("HybridVAE Training Summary", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path, "training_summary.png")


def plot_latent_space(
    model_path: Path,
    data_dir: Path,
    embeddings_path: Path,
    output_path: Path,
    n_samples: int = 2000,
) -> None:
    """Visualize learned latent space using t-SNE and PCA."""
    from src.ml.model import HybridVAE
    from src.preprocessing.embeddings import load_embeddings

    device = _get_device()
    logger.info(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})

    # Load embeddings
    embeddings_file = Path(embeddings_path)
    mappings_file = embeddings_file.with_name(f"{embeddings_file.stem}_mappings.pkl")
    embeddings, _, _ = load_embeddings(str(embeddings_file), str(mappings_file))

    # Create and load model
    model = HybridVAE(
        n_items=model_config.get("n_items", embeddings.shape[0]),
        item_embeddings=embeddings,
        latent_dim=model_config.get("latent_dim", 64),
        hidden_dims=model_config.get("hidden_dims", [256]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load interactions and sample users
    with open(data_dir / "interaction_matrix.pkl", "rb") as f:
        interaction_matrix = pickle.load(f)

    n_users = interaction_matrix.shape[0]
    indices = (
        np.random.choice(n_users, min(n_samples, n_users), replace=False)
        if n_users > n_samples
        else np.arange(n_users)
    )

    # Compute user embeddings
    user_embeddings, activity_levels = [], []
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing user embeddings"):
            user_vec = (
                torch.FloatTensor(interaction_matrix[idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )
            mu, _ = model.encode(user_vec)
            user_embeddings.append(mu.cpu().numpy().flatten())
            activity_levels.append(interaction_matrix[idx].nnz)

    user_embeddings = np.array(user_embeddings)
    log_activity = np.log1p(activity_levels)

    # Compute projections
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(user_embeddings)

    logger.info("Computing t-SNE projection...")
    tsne_result = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(
        user_embeddings
    )

    # Individual plots
    for name, result, labels in [
        (
            "latent_space_pca.png",
            pca_result,
            (
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
                "User Latent Space (PCA)",
            ),
        ),
        ("latent_space_tsne.png", tsne_result, ("t-SNE 1", "t-SNE 2", "User Latent Space (t-SNE)")),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            result[:, 0], result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
        )
        plt.colorbar(scatter, ax=ax, label="Log(1 + #Interactions)")
        ax.set(xlabel=labels[0], ylabel=labels[1], title=labels[2])
        plt.tight_layout()
        _save_figure(fig, output_path, name)

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, result, labels in [
        (
            axes[0],
            pca_result,
            (
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
                "PCA Projection",
            ),
        ),
        (axes[1], tsne_result, ("t-SNE 1", "t-SNE 2", "t-SNE Projection")),
    ]:
        scatter = ax.scatter(
            result[:, 0], result[:, 1], c=log_activity, cmap="viridis", alpha=0.6, s=20
        )
        ax.set(xlabel=labels[0], ylabel=labels[1], title=labels[2])
        plt.colorbar(scatter, ax=ax, label="Log(1 + #Interactions)")

    plt.suptitle("User Latent Space Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, "latent_space_combined.png")


# =============================================================================
# Baseline Visualizations
# =============================================================================


def plot_baseline_comparison(
    all_results: dict[str, dict],
    k_values: list[int],
    output_path: Path,
    n_negatives: int | None = 99,
) -> None:
    """Generate baseline comparison visualizations."""
    plt.style.use(STYLE)
    models = list(all_results.keys())
    protocol = f"Negative Sampling ({n_negatives})" if n_negatives else "Full Ranking"
    x = np.arange(len(k_values))
    width = 0.25

    # Bar chart plots for each metric
    for metric, filename in [
        ("recall", "baseline_recall_comparison.png"),
        ("ndcg", "baseline_ndcg_comparison.png"),
        ("hit_ratio", "baseline_hitrate_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model in enumerate(models):
            values = [all_results[model][k][metric] for k in k_values]
            ax.bar(x + width * i, values, width, label=model, color=COLORS[i % len(COLORS)])
        ax.set(
            xlabel="K",
            ylabel=f"{metric.replace('_', ' ').title()}@K",
            title=f"{metric.replace('_', ' ').title()}@K Comparison ({protocol})",
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"@{k}" for k in k_values])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        _save_figure(fig, output_path, filename)

    # Combined summary (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    k_main = 10

    for ax, metric, title in [
        (axes[0, 0], "recall", "Recall@10"),
        (axes[0, 1], "ndcg", "NDCG@10"),
        (axes[1, 0], "hit_ratio", "Hit Ratio@10"),
    ]:
        values = [all_results[m][k_main][metric] for m in models]
        bars = ax.bar(models, values, color=COLORS[: len(models)])
        ax.set(ylabel=title, title=f"{title} by Model")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Line plot
    for i, model in enumerate(models):
        recalls = [all_results[model][k]["recall"] for k in k_values]
        axes[1, 1].plot(
            k_values,
            recalls,
            "o-",
            label=model,
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            markersize=8,
        )
    axes[1, 1].set(xlabel="K", ylabel="Recall@K", title="Recall@K Trend")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Baseline Models Comparison ({protocol})", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path, "baseline_summary.png")


# =============================================================================
# Grid Search Visualizations
# =============================================================================


def plot_grid_search_results(results_path: Path, output_path: Path) -> None:
    """Generate grid search visualizations."""
    plt.style.use(STYLE)

    with open(results_path) as f:
        data = json.load(f)

    all_results = [r for r in data["all_results"] if "error" not in r]
    if not all_results:
        logger.error("No valid results found in grid search")
        return

    configs = [r["config"] for r in all_results]
    ndcg_scores = [r["ndcg@10"] for r in all_results]
    recall_scores = [r["recall@10"] for r in all_results]

    # Plot 1: Top 10 configurations
    sorted_results = sorted(all_results, key=lambda r: r["ndcg@10"], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"Config {i+1}" for i in range(len(sorted_results))]
    x = np.arange(len(labels))
    width = 0.35

    bars = ax.bar(
        x - width / 2,
        [r["ndcg@10"] for r in sorted_results],
        width,
        label="NDCG@10",
        color="#3498db",
    )
    ax.bar(
        x + width / 2,
        [r["recall@10"] for r in sorted_results],
        width,
        label="Recall@10",
        color="#2ecc71",
    )
    ax.set(
        xlabel="Configuration Rank", ylabel="Score", title="Top 10 Hyperparameter Configurations"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    _save_figure(fig, output_path, "grid_search_top_configs.png")

    # Plot 2: Parameter impact (box plots)
    param_names = list(configs[0].keys())
    n_params = min(len(param_names), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(param_names[:n_params]):
        param_groups: dict[str, list[float]] = {}
        for cfg, ndcg in zip(configs, ndcg_scores):
            key = str(cfg[param])
            param_groups.setdefault(key, []).append(ndcg)

        group_labels = list(param_groups.keys())
        bp = axes[idx].boxplot(
            [param_groups[lbl] for lbl in group_labels], tick_labels=group_labels, patch_artist=True
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)
        axes[idx].set(title=param, ylabel="NDCG@10")
        axes[idx].grid(True, alpha=0.3, axis="y")
        axes[idx].tick_params(axis="x", rotation=45)

    for idx in range(n_params, 6):
        axes[idx].set_visible(False)

    plt.suptitle("Hyperparameter Impact on NDCG@10", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path, "grid_search_param_impact.png")

    # Plot 3: Heatmap (latent_dim vs beta)
    if "latent_dim" in param_names and "beta" in param_names:
        latent_dims = sorted(set(c["latent_dim"] for c in configs))
        betas = sorted(set(c["beta"] for c in configs))
        heatmap = np.zeros((len(latent_dims), len(betas)))
        counts = np.zeros_like(heatmap)

        for cfg, ndcg in zip(configs, ndcg_scores):
            i, j = latent_dims.index(cfg["latent_dim"]), betas.index(cfg["beta"])
            heatmap[i, j] += ndcg
            counts[i, j] += 1

        heatmap /= np.maximum(counts, 1)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(len(betas)))
        ax.set_yticks(np.arange(len(latent_dims)))
        ax.set_xticklabels([str(b) for b in betas])
        ax.set_yticklabels([str(ld) for ld in latent_dims])
        ax.set(
            xlabel="Beta (KL Weight)",
            ylabel="Latent Dimension",
            title="Average NDCG@10: Latent Dim vs Beta",
        )

        for i in range(len(latent_dims)):
            for j in range(len(betas)):
                ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center", fontsize=10)

        plt.colorbar(im, ax=ax, label="NDCG@10")
        plt.tight_layout()
        _save_figure(fig, output_path, "grid_search_heatmap.png")

    # Plot 4: Scatter with best config
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(recall_scores, ndcg_scores, alpha=0.6, s=50, c="#3498db")

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
    ax.set(xlabel="Recall@10", ylabel="NDCG@10", title="Grid Search Results: Recall vs NDCG")
    ax.legend()
    ax.grid(True, alpha=0.3)

    annotation = "\n".join(f"{k}: {v}" for k, v in data["best_config"].items())
    ax.annotate(
        f"Best:\n{annotation}",
        xy=(recall_scores[best_idx], ndcg_scores[best_idx]),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="yellow", alpha=0.5),
    )

    plt.tight_layout()
    _save_figure(fig, output_path, "grid_search_scatter.png")


# =============================================================================
# CLI Entry Points
# =============================================================================


def visualize_training(model_dir: str, data_dir: str, embeddings_path: str) -> None:
    """Generate training visualizations from saved history."""
    model_path = Path(model_dir)
    figures_path = model_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    history_path = model_path / "training_history.json"
    if not history_path.exists():
        logger.error(f"Training history not found: {history_path}. Run `make train` first.")
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

    best_model = model_path / "best_model.pth"
    if best_model.exists():
        logger.info("Generating latent space visualizations...")
        plot_latent_space(best_model, Path(data_dir), Path(embeddings_path), figures_path)
    else:
        logger.warning(f"Model not found: {best_model}, skipping latent space plots")

    logger.info(f"All visualizations saved to {figures_path}")


def visualize_baselines(model_dir: str) -> None:
    """Generate baseline comparison visualizations."""
    figures_path = Path(model_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    results_path = figures_path / "baseline_results.json"
    if not results_path.exists():
        logger.error(f"Baseline results not found: {results_path}. Run `make baseline` first.")
        return

    with open(results_path) as f:
        raw = json.load(f)

    results = {model: {int(k): v for k, v in metrics.items()} for model, metrics in raw.items()}
    k_values = sorted(next(iter(results.values())).keys())

    logger.info("Generating baseline comparison visualizations...")
    plot_baseline_comparison(results, k_values, figures_path)
    logger.info(f"All visualizations saved to {figures_path}")


def visualize_grid_search(model_dir: str) -> None:
    """Generate grid search visualizations."""
    figures_path = Path(model_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(model_dir) / "grid_search_results.json"
    if not results_path.exists():
        logger.error(f"Grid search results not found: {results_path}. Run `make tune` first.")
        return

    logger.info("Generating grid search visualizations...")
    plot_grid_search_results(results_path, figures_path)
    logger.info(f"All visualizations saved to {figures_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for recommendation system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Visualization type")

    # Subcommands with shared defaults
    defaults = {
        "model_dir": str(config.MODEL_DIR),
        "data_dir": str(config.DATA_DIR),
        "embeddings": str(config.EMBEDDINGS_FILE),
    }

    train_p = subparsers.add_parser("training", help="Generate training visualizations")
    train_p.add_argument("--model-dir", default=defaults["model_dir"])
    train_p.add_argument("--data-dir", default=defaults["data_dir"])
    train_p.add_argument("--embeddings", default=defaults["embeddings"])

    baseline_p = subparsers.add_parser("baseline", help="Generate baseline visualizations")
    baseline_p.add_argument("--model-dir", default=defaults["model_dir"])

    tune_p = subparsers.add_parser("tuning", help="Generate grid search visualizations")
    tune_p.add_argument("--model-dir", default=defaults["model_dir"])

    all_p = subparsers.add_parser("all", help="Generate all visualizations")
    all_p.add_argument("--model-dir", default=defaults["model_dir"])
    all_p.add_argument("--data-dir", default=defaults["data_dir"])
    all_p.add_argument("--embeddings", default=defaults["embeddings"])

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
