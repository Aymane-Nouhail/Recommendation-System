.PHONY: clean download preprocess train evaluate run-api all visualize visualize-training visualize-baseline visualize-tuning tune

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
all: preprocess train evaluate baseline visualize

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
	$(PYTHON) $(SRC_DIR)/ml/train.py --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --output $(MODELS_DIR)/ --epochs 20 --latent-dim 128 --hidden-dims 512 --dropout 0.3 --beta 0.1 --learning-rate 0.001

# Evaluate the model (negative sampling protocol - default 99 negatives)
evaluate:
	@echo "Evaluating model with negative sampling protocol (99 negatives)..."
	$(PYTHON) $(SRC_DIR)/ml/evaluate.py --model $(MODELS_DIR)/best_model.pth --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --n-negatives 99 --output $(MODELS_DIR)/figures/evaluation_results.json

# Evaluate with full ranking protocol (harder)
evaluate-full:
	@echo "Evaluating model with full ranking protocol (all items)..."
	$(PYTHON) $(SRC_DIR)/ml/evaluate.py --model $(MODELS_DIR)/best_model.pth --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --n-negatives 0 --output $(MODELS_DIR)/figures/evaluation_full_results.json

# Run all baseline models with negative sampling (Popularity, Item-KNN, SVD)
baseline:
	@echo "Running all baseline models with negative sampling (99 negatives)..."
	$(PYTHON) $(SRC_DIR)/ml/baseline.py --data $(DATA_DIR)/ --model all --n-negatives 99

# Run all baseline models with full ranking
baseline-full:
	@echo "Running all baseline models with full ranking..."
	$(PYTHON) $(SRC_DIR)/ml/baseline.py --data $(DATA_DIR)/ --model all --n-negatives 0

# Generate all visualizations (training + baseline + tuning)
visualize: visualize-training visualize-baseline visualize-tuning

# Generate training visualizations (loss curves, latent space)
visualize-training:
	@echo "Generating training visualizations..."
	$(PYTHON) $(SRC_DIR)/ml/visualize.py training --model-dir $(MODELS_DIR)/ --data-dir $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy

# Generate baseline comparison visualizations
visualize-baseline:
	@echo "Generating baseline comparison visualizations..."
	$(PYTHON) $(SRC_DIR)/ml/visualize.py baseline --model-dir $(MODELS_DIR)/

# Generate grid search / tuning visualizations
visualize-tuning:
	@echo "Generating grid search visualizations..."
	$(PYTHON) $(SRC_DIR)/ml/visualize.py tuning --model-dir $(MODELS_DIR)/

# Run the API server
run-api:
	@echo "Starting API server..."
	$(PYTHON) $(SRC_DIR)/api/server.py

# Hyperparameter tuning (grid search)
tune:
	@echo "Running hyperparameter grid search..."
	$(PYTHON) $(SRC_DIR)/ml/tune.py --data $(DATA_DIR)/ --embeddings $(EMBEDDINGS_DIR)/item_embeddings.npy --output $(MODELS_DIR)/ --epochs 10 --patience 3

