from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to generate recommendations for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    exclude_seen: bool = Field(
        default=True, description="Whether to exclude items the user has already seen"
    )


class RecommendationItem(BaseModel):
    item_id: str = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")


class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    total_items: int = Field(..., description="Total number of items in the catalog")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_users: int
    total_items: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str = Field(..., description="User ID")
    interaction_count: int = Field(..., description="Number of interactions")


class UsersResponse(BaseModel):
    users: List[UserInfo] = Field(..., description="List of users with interaction counts")
    total: int = Field(..., description="Total number of users")
    returned: int = Field(..., description="Number of users returned")
