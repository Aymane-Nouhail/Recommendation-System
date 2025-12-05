.PHONY: clean download preprocess train evaluate run-api all

# Load environment variables
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Variables
PYTHON = python
DATA_DIR = data
MODELS_DIR = models
EMBEDDINGS_DIR = embeddings
SRC_DIR = src

# Default targets
all: clean preprocess train evaluate baseline

# Clean all generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(DATA_DIR)/cleaned_reviews.jsonl
	rm -rf $(DATA_DIR)/processed_dataset
	rm -rf $(DATA_DIR)/*.csv
	rm -rf $(DATA_DIR)/*.pkl
	rm -rf $(EMBEDDINGS_DIR)/*
	rm -rf $(MODELS_DIR)/*
	@echo "Clean complete."

# Download dataset
download:
	@echo "Downloading dataset..."
	$(PYTHON) download_data.py

# Preprocess data pipeline
preprocess:
	@echo "Step 1: Cleaning raw data..."
	$(PYTHON) $(SRC_DIR)/preprocessing/cleaning.py --input $(DATA_DIR)/$(RAW_DATA_FILE) --output $(DATA_DIR)/cleaned_reviews.jsonl
	
	@echo "Step 2: Building dataset splits..."
	$(PYTHON) $(SRC_DIR)/preprocessing/dataset.py --input $(DATA_DIR)/cleaned_reviews.jsonl --output $(DATA_DIR)/
	
	@echo "Step 3: Computing item embeddings..."
	$(PYTHON) $(SRC_DIR)/preprocessing/embeddings.py --input $(DATA_DIR)/cleaned_reviews.jsonl --output $(EMBEDDINGS_DIR)/item_embeddings.npy --model $(EMBEDDING_MODEL)

# Train the model
train:
	@echo "Training model..."
	$(PYTHON) $(SRC_DIR)/ml/train.py --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --output $(MODELS_DIR)/ --epochs 20 --latent-dim 64 --hidden-dims 256 --use-annealing --dropout 0.5

# Evaluate the model (negative sampling protocol - default 99 negatives)
evaluate:
	@echo "Evaluating model with negative sampling protocol (99 negatives)..."
	$(PYTHON) $(SRC_DIR)/ml/evaluate.py --model $(MODELS_DIR)/best_model.pth --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --n-negatives 99

# Evaluate with full ranking protocol (harder)
evaluate-full:
	@echo "Evaluating model with full ranking protocol (all items)..."
	$(PYTHON) $(SRC_DIR)/ml/evaluate.py --model $(MODELS_DIR)/best_model.pth --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --n-negatives 0

# Run all baseline models with negative sampling (Popularity, Item-KNN, SVD)
baseline:
	@echo "Running all baseline models with negative sampling (99 negatives)..."
	$(PYTHON) $(SRC_DIR)/ml/baseline.py --data $(DATA_DIR)/ --model all --n-negatives 99

# Run all baseline models with full ranking
baseline-full:
	@echo "Running all baseline models with full ranking..."
	$(PYTHON) $(SRC_DIR)/ml/baseline.py --data $(DATA_DIR)/ --model all --n-negatives 0

# Run the API server
run-api:
	@echo "Starting API server..."
	$(PYTHON) $(SRC_DIR)/api/server.py
