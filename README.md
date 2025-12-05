# Hybrid Variational Autoencoder for Recommendation# HybridVAE Recommendation System# HybridVAE Recommendation System# Hybrid VAE Recommendation System



This repository contains an implementation of a hybrid recommendation system that combines collaborative filtering with pre-trained text embeddings. The model uses a Variational Autoencoder (VAE) architecture where item embeddings from Sentence-BERT serve as a frozen decoder, enabling the system to leverage both user-item interaction patterns and semantic item representations.



The project includes a fully autonomous pipeline that handles the entire workflow from raw data to evaluation results with a single command.A state-of-the-art recommendation system using a Hybrid Variational Autoencoder that combines collaborative filtering with semantic item embeddings from SBERT. Evaluated on Amazon product review datasets.



---



## Table of Contents## 🎯 Key ResultsA state-of-the-art recommendation system using a Hybrid Variational Autoencoder that combines collaborative filtering with semantic item embeddings from SBERT. Evaluated on Amazon product review datasets.A sophisticated recommendation system using a Hybrid Variational Autoencoder (VAE) that combines collaborative filtering with item text embeddings from SBERT.



- [Results](#results)

- [Model Architecture](#model-architecture)

- [Autonomous Pipeline](#autonomous-pipeline)### Performance Comparison (NDCG@10)

- [Implementation Details](#implementation-details)

- [Installation](#installation)

- [Usage](#usage)

- [Methodology](#methodology)| Model | All_Beauty | Appliances |## 🎯 Key Results## Project Structure

- [Project Structure](#project-structure)

- [References](#references)|-------|------------|------------|



---| **HybridVAE (Ours)** | **0.213** | 0.663 |



## Results| LightGCN | 0.189 | **0.668** |



The model was evaluated on two Amazon product review datasets with different characteristics.| SVD | 0.189 | 0.373 |### Performance Comparison (NDCG@10)```



### Summary (NDCG@10)| Mult-VAE | 0.178 | 0.529 |



| Model | All_Beauty | Appliances || Item-KNN | 0.152 | 0.317 |recommendation_system/

|:------|:----------:|:----------:|

| **HybridVAE** | **0.213** | 0.663 || Popularity | 0.104 | 0.185 |

| LightGCN | 0.189 | **0.668** |

| SVD | 0.189 | 0.373 || Model | All_Beauty | Appliances |├── data/                    # Dataset storage

| Mult-VAE | 0.178 | 0.529 |

| Item-KNN | 0.152 | 0.317 |**Key Findings:**

| Popularity | 0.104 | 0.185 |

- HybridVAE outperforms LightGCN by **12.7%** on All_Beauty (sparse dataset)|-------|------------|------------|├── models/                  # Saved model checkpoints

On the sparse All_Beauty dataset (22,363 users, 12,101 items), HybridVAE outperforms LightGCN by 12.7%. On the denser Appliances dataset (2,072 users, 890 items), the two models perform comparably.

- Competitive with LightGCN on Appliances (dense dataset)

### Baseline Comparison

- Significantly outperforms traditional methods (SVD, Item-KNN, Popularity)| **HybridVAE (Ours)** | **0.213** | 0.663 |├── embeddings/             # Pre-computed item embeddings

<p align="center">

  <img src="assets/all_beauty_baseline.png" width="45%" />

  &nbsp;&nbsp;

  <img src="assets/appliances_baseline.png" width="45%" />### Baseline Comparison| LightGCN | 0.189 | **0.668** |├── src/                    # Source code

</p>



<p align="center">

  <sub>Left: All_Beauty dataset. Right: Appliances dataset.</sub><p align="center">| SVD | 0.189 | 0.373 |│   ├── preprocessing/      # Data processing modules

</p>

  <img src="assets/all_beauty_baseline.png" width="48%" alt="All_Beauty Baseline Comparison"/>

### Detailed Results

  <img src="assets/appliances_baseline.png" width="48%" alt="Appliances Baseline Comparison"/>| Mult-VAE | 0.178 | 0.529 |│   │   ├── cleaning.py     # Data loading and cleaning

<details>

<summary>All_Beauty Dataset</summary></p>



| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |<p align="center"><em>Left: All_Beauty dataset | Right: Appliances dataset</em></p>| Item-KNN | 0.152 | 0.317 |│   │   ├── dataset.py      # Dataset construction

|:------|:--------:|:---------:|:---------:|:------:|:-------:|:-------:|

| HybridVAE | 0.195 | 0.286 | 0.403 | 0.181 | 0.213 | 0.247 |

| LightGCN | 0.161 | 0.240 | 0.353 | 0.158 | 0.189 | 0.224 |

| SVD | 0.160 | 0.243 | 0.365 | 0.156 | 0.189 | 0.227 |### Full Results| Popularity | 0.104 | 0.185 |│   │   └── embeddings.py   # SBERT text embeddings

| Mult-VAE | 0.149 | 0.227 | 0.342 | 0.145 | 0.178 | 0.214 |

| Item-KNN | 0.117 | 0.189 | 0.299 | 0.120 | 0.152 | 0.189 |

| Popularity | 0.076 | 0.131 | 0.219 | 0.080 | 0.104 | 0.134 |

#### All_Beauty Dataset (22,363 users, 12,101 items)│   ├── ml/                 # Machine learning modules

</details>

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |

<details>

<summary>Appliances Dataset</summary>|-------|----------|-----------|-----------|--------|---------|---------|### Full Results│   │   ├── model.py        # Hybrid VAE implementation



| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 || **HybridVAE** | 0.195 | **0.286** | 0.403 | **0.181** | **0.213** | 0.247 |

|:------|:--------:|:---------:|:---------:|:------:|:-------:|:-------:|

| HybridVAE | 0.603 | 0.711 | 0.810 | 0.600 | 0.663 | 0.688 || LightGCN | 0.161 | 0.240 | 0.353 | 0.158 | 0.189 | 0.224 |│   │   ├── train.py        # Training loop

| LightGCN | 0.617 | 0.724 | 0.820 | 0.605 | 0.668 | 0.694 |

| SVD | 0.386 | 0.482 | 0.609 | 0.341 | 0.373 | 0.413 || SVD | 0.160 | 0.243 | 0.365 | 0.156 | 0.189 | 0.227 |

| Mult-VAE | 0.471 | 0.583 | 0.704 | 0.478 | 0.529 | 0.563 |

| Item-KNN | 0.265 | 0.367 | 0.502 | 0.267 | 0.317 | 0.362 |#### All_Beauty Dataset (22,363 users, 12,101 items)│   │   └── evaluate.py     # Evaluation metrics

| Popularity | 0.149 | 0.224 | 0.342 | 0.147 | 0.185 | 0.226 |

#### Appliances Dataset (2,072 users, 890 items)

</details>

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 || Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |│   ├── api/                # API modules

---

|-------|----------|-----------|-----------|--------|---------|---------|

## Model Architecture

| **HybridVAE** | 0.603 | 0.711 | 0.810 | 0.600 | 0.663 | 0.688 ||-------|----------|-----------|-----------|--------|---------|---------|│   │   ├── server.py       # FastAPI server

The HybridVAE combines a standard VAE encoder with a decoder that uses pre-computed Sentence-BERT embeddings as fixed weights.

| LightGCN | 0.617 | 0.724 | 0.820 | 0.605 | 0.668 | 0.694 |

```

Input: User interaction vector x ∈ {0,1}^n_items| SVD | 0.386 | 0.482 | 0.609 | 0.341 | 0.373 | 0.413 || **HybridVAE** | 0.195 | **0.286** | 0.403 | **0.181** | **0.213** | 0.247 |│   │   └── schemas.py      # Pydantic models

                    │

                    ▼

┌───────────────────────────────────────────────┐

│ Encoder                                       │## 🏗️ Architecture| LightGCN | 0.161 | 0.240 | 0.353 | 0.158 | 0.189 | 0.224 |│   └── utils.py           # Helper functions

│   Linear(n_items → hidden_dim)                │

│   LayerNorm → GELU → Dropout                  │

│   Linear(hidden_dim → latent_dim) → μ         │

│   Linear(hidden_dim → latent_dim) → log(σ²)   │### HybridVAE Model| SVD | 0.160 | 0.243 | 0.365 | 0.156 | 0.189 | 0.227 |├── notebooks/              # Jupyter notebooks

└───────────────────────────────────────────────┘

                    │

                    ▼ z = μ + σ ⊙ ε,  ε ~ N(0, I)

                    │```├── requirements.txt        # Python dependencies

┌───────────────────────────────────────────────┐

│ Decoder                                       │Input: User interaction vector (n_items,)

│   Projection: Linear(latent_dim → emb_dim)    │

│   Scores: E · z_projected  (E is frozen)      │         │#### Appliances Dataset (2,072 users, 890 items)└── README.md              # This file

└───────────────────────────────────────────────┘

                    │         ▼

                    ▼

Output: Reconstruction scores ∈ R^n_items┌─────────────────────────────────────┐| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |```

```

│  Encoder                            │

The item embedding matrix E is computed once using Sentence-BERT (`all-MiniLM-L6-v2`, 384 dimensions) on concatenated item titles and review text. These embeddings remain frozen during training, which serves as a regularization mechanism and enables the model to generalize to items with limited interaction data.

│  Linear(n_items → 512) + LayerNorm  │|-------|----------|-----------|-----------|--------|---------|---------|

**Loss function:**

│  GELU + Dropout(0.3)                │

```

L = L_recon + β · D_KL(q(z|x) || p(z))│  Linear(512 → μ, σ)                 │| **HybridVAE** | 0.603 | 0.711 | 0.810 | 0.600 | 0.663 | 0.688 |## Installation

```

└─────────────────────────────────────┘

where L_recon is the multinomial negative log-likelihood and β is annealed from 0 to 0.2 over training.

         │| LightGCN | 0.617 | 0.724 | 0.820 | 0.605 | 0.668 | 0.694 |

---

         ▼ Reparameterization: z = μ + σ·ε

## Autonomous Pipeline

         │| SVD | 0.386 | 0.482 | 0.609 | 0.341 | 0.373 | 0.413 |1. Clone the repository and navigate to the project directory:

A key feature of this project is the fully automated pipeline that executes the entire experimental workflow with a single command. The pipeline is implemented via a Makefile and handles:

┌─────────────────────────────────────┐

1. **Data preprocessing** — cleaning, filtering, and temporal train/validation/test splitting

2. **Embedding computation** — generating Sentence-BERT embeddings for all items│  Decoder (Frozen SBERT Embeddings)  │```bash

3. **Hyperparameter tuning** — grid search over latent dimensions, hidden dimensions, dropout, and β

4. **Model training** — training with the best configuration found during tuning│  Projection: Linear(latent → 384)   │

5. **Evaluation** — computing Recall@K, NDCG@K, and Hit Ratio@K on the test set

6. **Baseline comparison** — running Popularity, Item-KNN, SVD, Mult-VAE, and LightGCN│  logits = Embeddings @ z            │**Key Findings:**cd recommendation_system

7. **Visualization** — generating training curves, baseline comparisons, and latent space plots

└─────────────────────────────────────┘

To run the complete pipeline:

         │- HybridVAE outperforms LightGCN by **12.7%** on All_Beauty (sparse dataset)```

```bash

make all         ▼

```

Output: Reconstruction scores (n_items,)- Competitive with LightGCN on Appliances (dense dataset)

The pipeline reads the dataset configuration from `.env` and automatically propagates the best hyperparameters from tuning to training. Results are saved to JSON files for reproducibility, and all generated plots are stored in `models/figures/`.

```

### Pipeline Implementation

- Significantly outperforms traditional methods (SVD, Item-KNN, Popularity)2. Install dependencies:

The `train-best` target demonstrates how the pipeline chains steps together:

**Key Components:**

```makefile

train-best:- **SBERT Embeddings**: `all-MiniLM-L6-v2` (384 dimensions) - frozen during training```bash

    @# Extract best config from grid search results

    @$(PYTHON) -c "\- **Latent Space**: 128 dimensions (found via grid search)

    import json; \

    cfg = json.load(open('models/grid_search_results.json'))['best_config']; \- **Loss**: Reconstruction (BCE) + β × KL Divergence with annealing## 🏗️ Architecturepip install -r requirements.txt

    ..." 

    @# Train with extracted hyperparameters- **Regularization**: Dropout, LayerNorm, gradient clipping

    $(PYTHON) src/ml/train.py --latent-dim $(LATENT) --hidden-dims $(HIDDEN) ...

``````



This design ensures that:## 📈 Training & Tuning

- No manual intervention is required between steps

- Hyperparameters flow automatically from tuning to training### HybridVAE Model

- Experiments are fully reproducible from the same `.env` configuration

### Training Curves

---

## Dataset Format

## Implementation Details

<p align="center">

### Data Preprocessing (`src/preprocessing/`)

  <img src="assets/all_beauty_training.png" width="48%" alt="All_Beauty Training"/>```

- **cleaning.py**: Loads JSONL Amazon review data, filters users and items with fewer than 5 interactions, and converts ratings ≥4 to positive interactions.

- **dataset.py**: Creates temporal train/validation/test splits (70/15/15) based on review timestamps. Builds the user-item interaction matrix and saves mappings.  <img src="assets/appliances_training.png" width="48%" alt="Appliances Training"/>

- **embeddings.py**: Generates item embeddings by concatenating the product title with aggregated review text, then encoding with Sentence-BERT.

</p>Input: User interaction vector (n_items,)The system expects Amazon Reviews data in JSONL format with the following fields:

### Model (`src/ml/model.py`)

<p align="center"><em>Training loss curves and metrics over epochs</em></p>

The `HybridVAE` class implements:

- Configurable encoder with LayerNorm and GELU activations         │- `user_id`: Unique user identifier

- Reparameterization trick for sampling from the latent distribution

- Projection layer mapping latent space to embedding space### Grid Search Hyperparameter Tuning

- KL annealing schedule to prevent posterior collapse

- Frozen embedding decoder         ▼- `asin`: Amazon Standard Identification Number (item ID)



### Training (`src/ml/train.py`)<p align="center">



Training features include:  <img src="assets/all_beauty_gridsearch.png" width="60%" alt="Grid Search Heatmap"/>┌─────────────────────────────────────┐- `rating`: Rating score (1-5)

- Adam optimizer with gradient clipping (max norm 5.0)

- Early stopping based on validation loss</p>

- Checkpoint saving at each epoch

- Training history export to JSON<p align="center"><em>Hyperparameter impact on NDCG@10 (All_Beauty)</em></p>│  Encoder                            │- `title`: Product title



### Hyperparameter Tuning (`src/ml/tune.py`)



Grid search over:### Latent Space Visualization│  Linear(n_items → 512) + LayerNorm  │- `text`: Review text

- `latent_dim`: [64, 128]

- `hidden_dims`: [[256], [512]]

- `dropout`: [0.3, 0.5]

- `beta`: [0.1, 0.2]<p align="center">│  GELU + Dropout(0.3)                │- `timestamp`: Review timestamp

- `learning_rate`: [1e-3]

  <img src="assets/all_beauty_latent_tsne.png" width="48%" alt="All_Beauty Latent Space"/>

Each configuration is trained for 5 epochs with early stopping (patience=2). The best configuration is selected based on NDCG@10 on the validation set.

  <img src="assets/appliances_latent_tsne.png" width="48%" alt="Appliances Latent Space"/>│  Linear(512 → μ, σ)                 │

### Evaluation (`src/ml/evaluate.py`)

</p>

Evaluation uses the standard negative sampling protocol:

- For each test interaction, sample 99 random negative items<p align="center"><em>t-SNE visualization of learned user embeddings</em></p>└─────────────────────────────────────┘## Usage

- Rank the test item among the 100 candidates

- Compute Recall@K, NDCG@K, and Hit Ratio@K for K ∈ {5, 10, 20}



### Baselines (`src/ml/baseline.py`)## 📁 Project Structure         │



| Model | Implementation |

|:------|:---------------|

| Popularity | Item frequency ranking |```         ▼ Reparameterization: z = μ + σ·ε### 1. Data Preprocessing

| Item-KNN | Cosine similarity on interaction vectors (k=50) |

| SVD | Matrix factorization via Surprise library (100 factors) |recommendation_system/

| Mult-VAE | Multinomial VAE following Liang et al. (2018) |

| LightGCN | 3-layer graph convolution with BPR loss |├── data/                    # Dataset storage         │```bash



All baselines use the same train/test splits and evaluation protocol for fair comparison.│   ├── All_Beauty.jsonl     # Raw Amazon reviews



---│   ├── train.csv            # Training split┌─────────────────────────────────────┐python src/preprocessing/cleaning.py --input data/reviews.jsonl --output data/cleaned_reviews.jsonl



## Installation│   ├── val.csv              # Validation split



```bash│   └── test.csv             # Test split│  Decoder (Frozen SBERT Embeddings)  │```

git clone https://github.com/Aymane-Nouhail/Recommendation-System.git

cd Recommendation-System├── embeddings/              # Pre-computed SBERT embeddings

pip install -r requirements.txt

```│   ├── item_embeddings.npy│  Projection: Linear(latent → 384)   │



Requirements include PyTorch, sentence-transformers, scipy, scikit-learn, and surprise.│   └── item_embeddings_mappings.pkl



---├── models/                  # Saved checkpoints & results│  logits = Embeddings @ z            │### 2. Build Dataset



## Usage│   ├── best_model.pth



### Full Pipeline│   ├── grid_search_results.json└─────────────────────────────────────┘```bash



```bash│   ├── evaluation_results.json

# Configure dataset in .env

echo "RAW_DATA_FILE=All_Beauty.jsonl" > .env│   └── baseline_results.json         │python src/preprocessing/dataset.py --input data/cleaned_reviews.jsonl --output data/



# Run everything├── assets/                  # README images

make all

```├── src/         ▼```



### Individual Steps│   ├── preprocessing/



```bash│   │   ├── cleaning.py      # Data loading & filteringOutput: Reconstruction scores (n_items,)

make preprocess    # Clean data, create splits, compute embeddings

make tune          # Grid search for best hyperparameters│   │   ├── dataset.py       # Train/val/test splits

make train-best    # Train with best configuration

make evaluate      # Evaluate on test set│   │   └── embeddings.py    # SBERT embedding generation```### 3. Compute Item Embeddings

make baseline      # Run all baseline models

make visualize     # Generate plots│   ├── ml/

```

│   │   ├── model.py         # HybridVAE architecture```bash

### Configuration

│   │   ├── train.py         # Training loop

Edit `.env` to change the dataset or model parameters:

│   │   ├── evaluate.py      # Metrics (Recall, NDCG, HR)**Key Components:**python src/preprocessing/embeddings.py --input data/cleaned_reviews.jsonl --output embeddings/item_embeddings.npy

```bash

RAW_DATA_FILE=All_Beauty.jsonl│   │   ├── tune.py          # Grid search hyperparameter tuning

EMBEDDING_MODEL=all-MiniLM-L6-v2

BATCH_SIZE=64│   │   ├── baseline.py      # Baseline models- **SBERT Embeddings**: `all-MiniLM-L6-v2` (384 dimensions) - frozen during training```

EPOCHS=20

```│   │   └── visualize.py     # Result visualization



---│   └── api/- **Latent Space**: 128 dimensions (found via grid search)



## Methodology│       ├── server.py        # FastAPI inference server



### Training Curves│       └── schemas.py       # Pydantic models- **Loss**: Reconstruction (BCE) + β × KL Divergence with annealing### 4. Train the Model



<p align="center">├── backups/                 # Experiment backups

  <img src="assets/all_beauty_training.png" width="45%" />

  &nbsp;&nbsp;├── Makefile                 # Pipeline automation- **Regularization**: Dropout, LayerNorm, gradient clipping```bash

  <img src="assets/appliances_training.png" width="45%" />

</p>├── requirements.txt



### Hyperparameter Search└── README.mdpython src/ml/train.py --data data/ --embeddings embeddings/item_embeddings.npy --output models/



<p align="center">```

  <img src="assets/all_beauty_gridsearch.png" width="60%" />

</p>## 📁 Project Structure```



<p align="center">## 🚀 Quick Start

  <sub>Impact of hyperparameters on NDCG@10 (All_Beauty dataset).</sub>

</p>



### Latent Space### Installation



<p align="center">```### 5. Evaluate the Model

  <img src="assets/all_beauty_latent_tsne.png" width="45%" />

  &nbsp;&nbsp;```bash

  <img src="assets/appliances_latent_tsne.png" width="45%" />

</p>git clone https://github.com/Aymane-Nouhail/Recommendation-System.gitrecommendation_system/```bash



<p align="center">cd recommendation_system

  <sub>t-SNE visualization of learned user representations.</sub>

</p>pip install -r requirements.txt├── data/                    # Dataset storagepython src/ml/evaluate.py --model models/best_model.pth --data data/ --embeddings embeddings/item_embeddings.npy



---```



## Project Structure│   ├── All_Beauty.jsonl     # Raw Amazon reviews```



```### Run Full Pipeline

recommendation_system/

├── data/                    # Raw and processed data│   ├── train.csv            # Training split

├── embeddings/              # Pre-computed item embeddings

├── models/                  # Checkpoints and results```bash

│   └── figures/             # Generated plots

├── assets/                  # README images# Run everything: preprocess → tune → train → evaluate → baselines → visualize│   ├── val.csv              # Validation split### 6. Start the API Server

├── src/

│   ├── preprocessing/make all

│   │   ├── cleaning.py

│   │   ├── dataset.py```│   └── test.csv             # Test split```bash

│   │   └── embeddings.py

│   ├── ml/

│   │   ├── model.py         # HybridVAE implementation

│   │   ├── train.py### Individual Steps├── embeddings/              # Pre-computed SBERT embeddingspython src/api/server.py

│   │   ├── evaluate.py

│   │   ├── tune.py          # Grid search

│   │   ├── baseline.py      # Baseline models

│   │   └── visualize.py```bash│   ├── item_embeddings.npy```

│   └── api/

│       ├── server.py        # FastAPI inference# 1. Preprocess data (clean, split, compute embeddings)

│       └── schemas.py

├── Makefile                 # Pipeline automationmake preprocess│   └── item_embeddings_mappings.pkl

├── requirements.txt

└── README.md

```

# 2. Hyperparameter tuning (grid search)├── models/                  # Saved checkpoints & resultsThe API will be available at `http://localhost:8000`

---

make tune

## References

│   ├── best_model.pth

- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.

- Liang, D., et al. (2018). Variational Autoencoders for Collaborative Filtering. *WWW*.# 3. Train with best hyperparameters

- He, X., et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. *SIGIR*.

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.make train-best│   ├── grid_search_results.json## API Endpoints



---



## License# 4. Evaluate HybridVAE│   ├── evaluation_results.json



MIT Licensemake evaluate


│   └── baseline_results.json### POST `/recommend`

# 5. Run baseline models

make baseline├── src/Get recommendations for a user.



# 6. Generate visualizations│   ├── preprocessing/

make visualize

```│   │   ├── cleaning.py      # Data loading & filtering**Request:**



## 🔬 Methodology│   │   ├── dataset.py       # Train/val/test splits```json



### Data Preprocessing│   │   └── embeddings.py    # SBERT embedding generation{

1. **Filtering**: Remove users with <5 interactions, items with <5 interactions

2. **Binary ratings**: Convert ratings ≥4 to positive (1), else negative (0)│   ├── ml/    "user_id": "user123",

3. **Temporal split**: Train (70%) / Validation (15%) / Test (15%) by timestamp

4. **Embeddings**: SBERT (`all-MiniLM-L6-v2`) on item title + aggregated reviews│   │   ├── model.py         # HybridVAE architecture    "top_k": 10



### Hyperparameter Tuning (Grid Search)│   │   ├── train.py         # Training loop}



| Parameter | Search Space | Best (All_Beauty) |│   │   ├── evaluate.py      # Metrics (Recall, NDCG, HR)```

|-----------|--------------|-------------------|

| latent_dim | [64, 128] | 128 |│   │   ├── tune.py          # Grid search hyperparameter tuning

| hidden_dims | [[256], [512]] | [512] |

| dropout | [0.3, 0.5] | 0.3 |│   │   ├── baseline.py      # Baseline models**Response:**

| beta | [0.1, 0.2] | 0.2 |

| learning_rate | [1e-3] | 1e-3 |│   │   └── visualize.py     # Result visualization```json



**Tuning Protocol:**│   └── api/{

- 5 epochs per config with early stopping (patience=2)

- Validation metric: NDCG@10│       ├── server.py        # FastAPI inference server    "user_id": "user123",

- Best config selected for final training (20 epochs)

│       └── schemas.py       # Pydantic models    "recommendations": [

### Evaluation Protocol

- **Negative Sampling**: 99 random negatives per test item├── backups/                 # Experiment backups        {

- **Metrics**: Recall@K, NDCG@K, Hit Ratio@K for K ∈ {5, 10, 20}

- **Fair comparison**: Same splits and protocol for all models├── Makefile                 # Pipeline automation            "item_id": "B001234567",



### Baseline Models├── requirements.txt            "score": 0.95

| Model | Description |

|-------|-------------|└── README.md        },

| Popularity | Rank by item frequency |

| Item-KNN | Cosine similarity on interaction vectors |```        ...

| SVD | Matrix factorization (surprise library) |

| Mult-VAE | Multinomial VAE (Liang et al., 2018) |    ]

| LightGCN | Graph convolution + BPR loss (He et al., 2020) |

## 🚀 Quick Start}

## ⚙️ Configuration

```

Environment variables (`.env`):

```bash### Installation

RAW_DATA_FILE=All_Beauty.jsonl

EMBEDDING_MODEL=all-MiniLM-L6-v2## Model Architecture

BATCH_SIZE=64

EPOCHS=20```bash

LATENT_DIM=128

```git clone https://github.com/Aymane-Nouhail/Recommendation-System.gitThe Hybrid VAE combines:



## 🔌 API Usagecd recommendation_system



Start the server:pip install -r requirements.txt1. **Collaborative Filtering**: User-item interaction patterns

```bash

make run-api```2. **Content-Based Filtering**: SBERT embeddings of item text (title + review)

# or

python src/api/server.py

```

### Run Full Pipeline### VAE Components:

Get recommendations:

```bash- **Encoder**: Maps user interaction vector to latent space (μ, σ)

curl -X POST http://localhost:8000/recommend \

  -H "Content-Type: application/json" \```bash- **Reparameterization**: z = μ + σ * ε (where ε ~ N(0,1))

  -d '{"user_id": "A123456", "top_k": 10}'

```# Run everything: preprocess → tune → train → evaluate → baselines → visualize- **Decoder**: Uses item embeddings as decoder weights: logits = E @ z



## 📚 Referencesmake all



- Kingma & Welling (2014). Auto-Encoding Variational Bayes```### Loss Function:

- Liang et al. (2018). Variational Autoencoders for Collaborative Filtering

- He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network```

- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

### Individual StepsLoss = Reconstruction Loss + β * KL Divergence

## 📝 License

```

MIT License

```bash

# 1. Preprocess data (clean, split, compute embeddings)## Evaluation Metrics

make preprocess

- **Recall@10**: Fraction of relevant items in top-10 recommendations

# 2. Hyperparameter tuning (grid search)- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10

make tune

## Configuration

# 3. Train with best hyperparameters

make train-bestKey hyperparameters can be adjusted in the training script:

- `latent_dim`: Dimensionality of latent space (default: 200)

# 4. Evaluate HybridVAE- `beta`: KL divergence weight (default: 0.2)

make evaluate- `learning_rate`: Adam optimizer learning rate (default: 0.001)

- `batch_size`: Mini-batch size (default: 512)

# 5. Run baseline models- `epochs`: Number of training epochs (default: 100)

make baseline

## License

# 6. Generate visualizations

make visualizeMIT License
```

## 🔬 Methodology

### Data Preprocessing
1. **Filtering**: Remove users with <5 interactions, items with <5 interactions
2. **Binary ratings**: Convert ratings ≥4 to positive (1), else negative (0)
3. **Temporal split**: Train (70%) / Validation (15%) / Test (15%) by timestamp
4. **Embeddings**: SBERT (`all-MiniLM-L6-v2`) on item title + aggregated reviews

### Hyperparameter Tuning (Grid Search)

| Parameter | Search Space | Best (All_Beauty) |
|-----------|--------------|-------------------|
| latent_dim | [64, 128] | 128 |
| hidden_dims | [[256], [512]] | [512] |
| dropout | [0.3, 0.5] | 0.3 |
| beta | [0.1, 0.2] | 0.2 |
| learning_rate | [1e-3] | 1e-3 |

**Tuning Protocol:**
- 5 epochs per config with early stopping (patience=2)
- Validation metric: NDCG@10
- Best config selected for final training (20 epochs)

### Evaluation Protocol
- **Negative Sampling**: 99 random negatives per test item
- **Metrics**: Recall@K, NDCG@K, Hit Ratio@K for K ∈ {5, 10, 20}
- **Fair comparison**: Same splits and protocol for all models

### Baseline Models
| Model | Description |
|-------|-------------|
| Popularity | Rank by item frequency |
| Item-KNN | Cosine similarity on interaction vectors |
| SVD | Matrix factorization (surprise library) |
| Mult-VAE | Multinomial VAE (Liang et al., 2018) |
| LightGCN | Graph convolution + BPR loss (He et al., 2020) |

## ⚙️ Configuration

Environment variables (`.env`):
```bash
RAW_DATA_FILE=All_Beauty.jsonl
EMBEDDING_MODEL=all-MiniLM-L6-v2
BATCH_SIZE=64
EPOCHS=20
LATENT_DIM=128
```

## 📊 Visualizations

The pipeline generates:
- `models/figures/training_curves.png` - Loss curves
- `models/figures/baseline_comparison.png` - Model comparison bar charts
- `models/figures/tuning_results.png` - Grid search heatmaps

## 🔌 API Usage

Start the server:
```bash
make run-api
# or
python src/api/server.py
```

Get recommendations:
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A123456", "top_k": 10}'
```

## 📚 References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes
- Liang et al. (2018). Variational Autoencoders for Collaborative Filtering
- He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

## 📝 License

MIT License
