# Hybrid VAE Recommendation System

A sophisticated recommendation system using a Hybrid Variational Autoencoder (VAE) that combines collaborative filtering with item text embeddings from SBERT.

## Project Structure

```
recommendation_system/
├── data/                    # Dataset storage
├── models/                  # Saved model checkpoints
├── embeddings/             # Pre-computed item embeddings
├── src/                    # Source code
│   ├── preprocessing/      # Data processing modules
│   │   ├── cleaning.py     # Data loading and cleaning
│   │   ├── dataset.py      # Dataset construction
│   │   └── embeddings.py   # SBERT text embeddings
│   ├── ml/                 # Machine learning modules
│   │   ├── model.py        # Hybrid VAE implementation
│   │   ├── train.py        # Training loop
│   │   └── evaluate.py     # Evaluation metrics
│   ├── api/                # API modules
│   │   ├── server.py       # FastAPI server
│   │   └── schemas.py      # Pydantic models
│   └── utils.py           # Helper functions
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd recommendation_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The system expects Amazon Reviews data in JSONL format with the following fields:
- `user_id`: Unique user identifier
- `asin`: Amazon Standard Identification Number (item ID)
- `rating`: Rating score (1-5)
- `title`: Product title
- `text`: Review text
- `timestamp`: Review timestamp

## Usage

### 1. Data Preprocessing
```bash
python src/preprocessing/cleaning.py --input data/reviews.jsonl --output data/cleaned_reviews.jsonl
```

### 2. Build Dataset
```bash
python src/preprocessing/dataset.py --input data/cleaned_reviews.jsonl --output data/
```

### 3. Compute Item Embeddings
```bash
python src/preprocessing/embeddings.py --input data/cleaned_reviews.jsonl --output embeddings/item_embeddings.npy
```

### 4. Train the Model
```bash
python src/ml/train.py --data data/ --embeddings embeddings/item_embeddings.npy --output models/
```

### 5. Evaluate the Model
```bash
python src/ml/evaluate.py --model models/best_model.pth --data data/ --embeddings embeddings/item_embeddings.npy
```

### 6. Start the API Server
```bash
python src/api/server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/recommend`
Get recommendations for a user.

**Request:**
```json
{
    "user_id": "user123",
    "top_k": 10
}
```

**Response:**
```json
{
    "user_id": "user123",
    "recommendations": [
        {
            "item_id": "B001234567",
            "score": 0.95
        },
        ...
    ]
}
```

## Model Architecture

The Hybrid VAE combines:

1. **Collaborative Filtering**: User-item interaction patterns
2. **Content-Based Filtering**: SBERT embeddings of item text (title + review)

### VAE Components:
- **Encoder**: Maps user interaction vector to latent space (μ, σ)
- **Reparameterization**: z = μ + σ * ε (where ε ~ N(0,1))
- **Decoder**: Uses item embeddings as decoder weights: logits = E @ z

### Loss Function:
```
Loss = Reconstruction Loss + β * KL Divergence
```

## Evaluation Metrics

- **Recall@10**: Fraction of relevant items in top-10 recommendations
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10

## Configuration

Key hyperparameters can be adjusted in the training script:
- `latent_dim`: Dimensionality of latent space (default: 200)
- `beta`: KL divergence weight (default: 0.2)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `batch_size`: Mini-batch size (default: 512)
- `epochs`: Number of training epochs (default: 100)

## License

MIT License