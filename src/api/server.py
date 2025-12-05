"""
FastAPI server for Hybrid VAE recommendation system.

This module provides a REST API for serving recommendations using the trained
Hybrid VAE model.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

# Add src to path
from api.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    UserInfo,
    UsersResponse,
)
from ml.evaluate import load_model_from_checkpoint
from ml.model import HybridVAE
from preprocessing.embeddings import load_embeddings
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and data
model: Optional[HybridVAE] = None
interaction_matrix = None
user_to_idx: Dict[str, int] = {}
item_to_idx: Dict[str, int] = {}
idx_to_item: Dict[int, str] = {}
device: torch.device = torch.device("cpu")

# FastAPI app
app = FastAPI(
    title="Hybrid VAE Recommendation API",
    description="REST API for serving recommendations using Hybrid Variational Autoencoder",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


# Dependency to check if model is loaded
def get_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        total_users=len(user_to_idx),
        total_items=len(item_to_idx),
    )


@app.get("/users", response_model=UsersResponse)
async def get_users(n: Optional[int] = 5):
    """
    Get list of available user IDs sorted by activity (most interactions first).

    Args:
        n: Number of users to return (default 5, None for all users)

    Returns:
        List of users with interaction counts, sorted by most active
    """
    # Get interaction counts per user from the interaction matrix
    user_interactions = []
    for user_id, user_idx in user_to_idx.items():
        if interaction_matrix is not None:
            count = int(interaction_matrix[user_idx].sum())
        else:
            count = 0
        user_interactions.append((user_id, count))

    # Sort by interaction count descending
    user_interactions.sort(key=lambda x: x[1], reverse=True)

    # Apply limit
    if n is not None:
        user_interactions = user_interactions[:n]

    users = [UserInfo(user_id=uid, interaction_count=cnt) for uid, cnt in user_interactions]

    return UsersResponse(
        users=users,
        total=len(user_to_idx),
        returned=len(users),
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest, vae_model: HybridVAE = Depends(get_model)
):
    """
    Generate recommendations for a user.

    Args:
        request: Recommendation request containing user_id and parameters

    Returns:
        List of recommended items with scores

    Raises:
        HTTPException: If user not found or other errors occur
    """
    try:
        # Check if user exists
        if request.user_id not in user_to_idx:
            raise HTTPException(
                status_code=404,
                detail=f"User '{request.user_id}' not found in training data",
            )

        user_idx = user_to_idx[request.user_id]

        # Get user interaction vector
        with torch.no_grad():
            user_vector = (
                torch.FloatTensor(interaction_matrix[user_idx].toarray().flatten())
                .unsqueeze(0)
                .to(device)
            )

            # Get user embedding
            user_embedding = vae_model.get_user_embedding(user_vector)

            # Get item scores
            scores = vae_model.decode(user_embedding)
            scores = scores.squeeze().cpu().numpy()

            # Exclude seen items if requested
            if request.exclude_seen:
                seen_items = interaction_matrix[user_idx].nonzero()[1]
                scores[seen_items] = -np.inf

            # Get top-k items
            top_indices = np.argsort(scores)[::-1][: request.top_k]
            top_scores = scores[top_indices]

            # Convert to response format
            recommendations = []
            for item_idx, score in zip(top_indices, top_scores):
                if item_idx in idx_to_item and not np.isinf(score):
                    recommendations.append(
                        RecommendationItem(item_id=idx_to_item[item_idx], score=float(score))
                    )

            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                total_items=len(item_to_idx),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """
    Get user profile information including interaction history.

    Args:
        user_id: User ID

    Returns:
        User profile information
    """
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    user_idx = user_to_idx[user_id]

    # Get user interactions
    user_interactions = interaction_matrix[user_idx].nonzero()[1]
    interacted_items = [
        idx_to_item[item_idx] for item_idx in user_interactions if item_idx in idx_to_item
    ]

    return {
        "user_id": user_id,
        "total_interactions": len(interacted_items),
        "interacted_items": interacted_items[:50],  # Limit to first 50 items
        "has_more_items": len(interacted_items) > 50,
    }


@app.get("/items/{item_id}/info")
async def get_item_info(item_id: str):
    """
    Get item information.

    Args:
        item_id: Item ID

    Returns:
        Item information
    """
    if item_id not in item_to_idx:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")

    item_idx = item_to_idx[item_id]

    # Count how many users interacted with this item
    item_interactions = (interaction_matrix[:, item_idx] > 0).sum()

    return {
        "item_id": item_id,
        "item_index": item_idx,
        "interaction_count": int(item_interactions),
        "popularity_rank": None,  # Could be computed if needed
    }


@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    total_interactions = interaction_matrix.nnz if interaction_matrix is not None else 0

    return {
        "total_users": len(user_to_idx),
        "total_items": len(item_to_idx),
        "total_interactions": int(total_interactions),
        "sparsity": (
            1 - (total_interactions / (len(user_to_idx) * len(item_to_idx)))
            if len(user_to_idx) > 0 and len(item_to_idx) > 0
            else 0
        ),
        "model_parameters": (
            sum(p.numel() for p in model.parameters()) if model is not None else 0
        ),
    }


# Batch recommendation endpoint for multiple users
@app.post("/recommend/batch")
async def get_batch_recommendations(
    user_ids: List[str],
    top_k: int = 10,
    exclude_seen: bool = True,
    vae_model: HybridVAE = Depends(get_model),
):
    """
    Generate recommendations for multiple users.

    Args:
        user_ids: List of user IDs
        top_k: Number of recommendations per user
        exclude_seen: Whether to exclude seen items

    Returns:
        Dictionary mapping user IDs to recommendations
    """
    if len(user_ids) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 100 users per batch request")

    results = {}

    for user_id in user_ids:
        try:
            request = RecommendationRequest(user_id=user_id, top_k=top_k, exclude_seen=exclude_seen)
            response = await get_recommendations(request, vae_model)
            results[user_id] = response.recommendations
        except HTTPException as e:
            results[user_id] = {"error": e.detail}
        except Exception:
            results[user_id] = {"error": "Internal server error"}

    return results


def load_model_and_data(
    model_path: str, data_dir: str, embeddings_path: str, device_name: str = "cpu"
) -> None:
    """
    Load model and data for the API.

    Args:
        model_path: Path to trained model
        data_dir: Directory containing processed dataset
        embeddings_path: Path to item embeddings
        device_name: Device to use ('cpu', 'cuda', or 'mps')
    """
    global model, interaction_matrix, user_to_idx, item_to_idx, idx_to_item, device

    logger.info("Loading model and data...")

    # Set device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        # Auto-detect best available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_path = Path(data_dir)

    # Load interaction matrix
    with open(data_path / "interaction_matrix.pkl", "rb") as f:
        interaction_matrix = pickle.load(f)

    # Load mappings
    with open(data_path / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]
    idx_to_item = mappings["idx_to_item"]

    # Load embeddings
    embeddings, _, _ = load_embeddings(embeddings_path)

    # Load model
    model = load_model_from_checkpoint(model_path, embeddings, device)
    model.eval()

    logger.info(f"Loaded model with {len(user_to_idx)} users and {len(item_to_idx)} items")
    logger.info("API ready to serve recommendations!")


def create_app(
    model_path: str, data_dir: str, embeddings_path: str, device_name: str = "cpu"
) -> FastAPI:
    """
    Create and configure the FastAPI app.

    Args:
        model_path: Path to trained model
        data_dir: Directory containing dataset
        embeddings_path: Path to embeddings
        device_name: Device to use

    Returns:
        Configured FastAPI app
    """
    load_model_and_data(model_path, data_dir, embeddings_path, device_name)
    return app


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run Hybrid VAE Recommendation API")
    parser.add_argument("--model", default=config.MODEL_FILE, help="Path to trained model")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="Directory containing dataset")
    parser.add_argument(
        "--embeddings", default=config.EMBEDDINGS_FILE, help="Path to item embeddings"
    )
    parser.add_argument("--host", default=config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to bind to")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for inference (auto detects best available)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    # Load model and data
    load_model_and_data(args.model, args.data, args.embeddings, args.device)

    # Run server
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
