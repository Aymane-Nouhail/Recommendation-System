"""
Hybrid Variational Autoencoder (VAE) for collaborative filtering with item embeddings.

This module implements a VAE that combines collaborative filtering with content-based
filtering using pre-computed item text embeddings from SBERT as decoder weights.

Architecture:
- Encoder: User interaction vector → latent representation (μ, log σ)
- Reparameterization: z = μ + σ * ε (where ε ~ N(0,1))
- Decoder: Item embeddings act as decoder weights: logits = E @ z
- Loss: Reconstruction loss + β * KL divergence
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridVAE(nn.Module):
    """
    Hybrid Variational Autoencoder for recommendation systems.

    Combines collaborative filtering (user-item interactions) with content-based
    filtering (item text embeddings) in a unified VAE framework.
    """

    def __init__(
        self,
        n_items: int,
        item_embeddings: np.ndarray,
        latent_dim: int = 200,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.5,
        beta: float = 0.2,
        freeze_embeddings: bool = True,
    ):
        """
        Initialize the Hybrid VAE model.

        Args:
            n_items: Number of items in the dataset
            item_embeddings: Pre-computed item embeddings (n_items x embedding_dim)
            latent_dim: Dimensionality of the latent space
            hidden_dims: List of hidden layer dimensions for encoder
            dropout: Dropout rate for regularization
            beta: Weight for KL divergence in the loss function
            freeze_embeddings: Whether to freeze SBERT embeddings (recommended)
        """
        super(HybridVAE, self).__init__()

        self.n_items = n_items
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta = beta

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [600, 200]

        self.hidden_dims = hidden_dims

        # Item embeddings - use buffer for frozen, parameter for trainable
        self.embedding_dim = item_embeddings.shape[1]
        if freeze_embeddings:
            self.register_buffer("item_embeddings", torch.FloatTensor(item_embeddings))
        else:
            self.item_embeddings = nn.Parameter(torch.FloatTensor(item_embeddings))

        logger.info("Initializing HybridVAE:")
        logger.info(f"  Items: {n_items}")
        logger.info(f"  Latent dimensions: {latent_dim}")
        logger.info(f"  Embedding dimensions: {self.embedding_dim}")
        logger.info(f"  Hidden dimensions: {hidden_dims}")
        logger.info(f"  Beta (KL weight): {beta}")
        logger.info(f"  Embeddings frozen: {freeze_embeddings}")

        # Build encoder layers
        self._build_encoder()

        # Projection layer: always exists (Identity if dimensions match)
        if self.latent_dim != self.embedding_dim:
            self.projection_layer = nn.Sequential(
                nn.Linear(self.latent_dim, self.embedding_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            logger.info(f"  Added projection MLP: {self.latent_dim} -> {self.embedding_dim}")
        else:
            self.projection_layer = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _build_encoder(self):
        """Build the encoder network with LayerNorm and GELU activation."""
        encoder_layers = []

        # Input dimension is number of items (user interaction vector)
        in_dim = self.n_items

        # Hidden layers with LayerNorm and GELU (modern architecture)
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),  # Modern activation, better than Tanh
                    nn.Dropout(self.dropout),
                ]
            )
            in_dim = hidden_dim

        # Output layers for mean and log variance
        self.encoder = nn.Sequential(*encoder_layers)

        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)

    def _init_weights(self):
        """Initialize model weights using He initialization for GELU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization works well with GELU
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode user interaction vector to latent parameters.

        Args:
            x: User interaction vector (batch_size x n_items)

        Returns:
            Tuple of (mu, logvar) tensors
        """
        # Pass through encoder
        h = self.encoder(x)

        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε

        Args:
            mu: Mean tensor (batch_size x latent_dim)
            logvar: Log variance tensor (batch_size x latent_dim)

        Returns:
            Latent sample z (batch_size x latent_dim)
        """
        if self.training:
            # Standard deviation from log variance
            std = torch.exp(0.5 * logvar)

            # Sample from standard normal distribution
            eps = torch.randn_like(std)

            # Reparameterization trick
            return mu + eps * std
        else:
            # During inference, use mean only
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to item scores using item embeddings.

        The key insight of the hybrid approach: item embeddings act as decoder weights.
        This allows the model to generalize to items with rich text descriptions.

        Args:
            z: Latent representation (batch_size x latent_dim)

        Returns:
            Item scores (batch_size x n_items)
        """
        # Project to embedding space (Identity if dimensions already match)
        user_embedding = self.projection_layer(z)

        # Compute scores as dot product: (batch_size x embedding_dim) @ (embedding_dim x n_items)
        scores = torch.matmul(user_embedding, self.item_embeddings.t())

        return scores

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: User interaction vector (batch_size x n_items)

        Returns:
            Tuple of (reconstructed_scores, mu, logvar)
        """
        # Encode
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode to item scores
        recon_scores = self.decode(z)

        return recon_scores, mu, logvar

    def get_user_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get user embedding for recommendation.

        Args:
            x: User interaction vector (batch_size x n_items)

        Returns:
            User embedding in latent space (batch_size x latent_dim)
        """
        mu, _ = self.encode(x)
        return mu  # Use mean as user representation

    def recommend(
        self, user_embedding: torch.Tensor, top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate recommendations for a user.

        Args:
            user_embedding: User embedding (1 x latent_dim)
            top_k: Number of items to recommend

        Returns:
            Tuple of (top_item_indices, top_scores)
        """
        with torch.no_grad():
            # Decode to get item scores
            scores = self.decode(user_embedding)

            # Get top-k items
            top_scores, top_indices = torch.topk(scores, top_k, dim=-1)

        return top_indices, top_scores


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: reconstruction loss + β * KL divergence.

    Args:
        recon_x: Reconstructed item scores (batch_size x n_items)
        x: Original interaction vector (batch_size x n_items)
        mu: Mean of latent distribution (batch_size x latent_dim)
        logvar: Log variance of latent distribution (batch_size x latent_dim)
        beta: Weight for KL divergence term

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss: Multinomial likelihood for ranking (implicit feedback)
    # Sum over items per user, then average over batch
    recon_loss = -torch.mean(torch.sum(x * F.log_softmax(recon_x, dim=-1), dim=-1))

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
    # Formula: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    # Sum over latent dimensions per user, then average over batch
    batch_size = x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # Total loss: weighted combination
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class AnnealedVAE(HybridVAE):
    """
    Hybrid VAE with annealed beta (KL weight scheduling).

    Gradually increases the KL weight during training to avoid posterior collapse.
    """

    def __init__(self, *args, **kwargs):
        # Extract annealing parameters
        self.beta_min = kwargs.pop("beta_min", 0.0)
        self.beta_max = kwargs.pop("beta_max", kwargs.get("beta", 0.2))
        self.anneal_steps = kwargs.pop("anneal_steps", 10000)

        super().__init__(*args, **kwargs)

        self.current_step = 0

    def get_current_beta(self) -> float:
        """Get current beta value based on annealing schedule."""
        if self.current_step >= self.anneal_steps:
            return self.beta_max

        # Linear annealing
        progress = self.current_step / self.anneal_steps
        return self.beta_min + progress * (self.beta_max - self.beta_min)

    def step_annealing(self):
        """Increment annealing step."""
        self.current_step += 1

    def compute_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss with current annealed beta."""
        current_beta = self.get_current_beta()
        return vae_loss_function(recon_x, x, mu, logvar, current_beta)


def create_hybrid_vae(
    n_items: int,
    item_embeddings: np.ndarray,
    latent_dim: int = 200,
    hidden_dims: Optional[list] = None,
    dropout: float = 0.5,
    beta: float = 0.2,
    use_annealing: bool = False,
    freeze_embeddings: bool = True,
    **annealing_kwargs,
) -> HybridVAE:
    """
    Factory function to create a Hybrid VAE model.

    Args:
        n_items: Number of items
        item_embeddings: Item embeddings matrix
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        beta: KL divergence weight
        use_annealing: Whether to use beta annealing
        freeze_embeddings: Whether to freeze SBERT embeddings (recommended)
        **annealing_kwargs: Additional arguments for annealing

    Returns:
        HybridVAE model instance
    """
    if use_annealing:
        return AnnealedVAE(
            n_items=n_items,
            item_embeddings=item_embeddings,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            beta=beta,
            freeze_embeddings=freeze_embeddings,
            **annealing_kwargs,
        )
    else:
        return HybridVAE(
            n_items=n_items,
            item_embeddings=item_embeddings,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            beta=beta,
            freeze_embeddings=freeze_embeddings,
        )
