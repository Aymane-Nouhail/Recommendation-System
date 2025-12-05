import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_FILE = os.getenv("RAW_DATA_FILE", "Toys_and_Games_5.jsonl")
    PROCESSED_DATA_FILE = os.getenv(
        "PROCESSED_DATA_FILE", str(DATA_DIR / "processed_interactions.csv")
    )

    # Model paths
    MODEL_DIR = BASE_DIR / "models"
    MODEL_FILE = os.getenv("MODEL_FILE", str(MODEL_DIR / "vae_model.pth"))
    ENCODER_FILE = os.getenv("ENCODER_FILE", str(MODEL_DIR / "label_encoders.pkl"))
    EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", str(MODEL_DIR / "item_embeddings.npy"))

    # Training hyperparameters
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
    EPOCHS = int(os.getenv("EPOCHS", 10))
    LATENT_DIM = int(os.getenv("LATENT_DIM", 50))
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", 256))

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    @classmethod
    def ensure_dirs(cls):
        """Ensure necessary directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_dirs()
