#!/bin/bash

# Hybrid VAE Recommendation System - Complete Pipeline Runner
# This script runs the complete pipeline from raw data to trained model

set -e  # Exit on any error

echo "🚀 Starting Hybrid VAE Recommendation System Pipeline"
echo "=================================================="

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading configuration from .env..."
    export $(grep -v '^#' .env | xargs)
fi

SRC_DIR="src"

echo "📊 Step 1: Data Preprocessing"
echo "=============================="
python "$SRC_DIR/preprocessing/cleaning.py"

if [ $? -eq 0 ]; then
    echo "✅ Preprocessing completed successfully"
else
    echo "❌ Preprocessing failed"
    exit 1
fi

echo ""
echo "🔧 Step 2: Dataset Building"
echo "==========================="
python "$SRC_DIR/preprocessing/dataset.py"

if [ $? -eq 0 ]; then
    echo "✅ Dataset building completed successfully"
else
    echo "❌ Dataset building failed"
    exit 1
fi

echo ""
echo "🤖 Step 3: Computing Item Embeddings"
echo "===================================="
python "$SRC_DIR/preprocessing/embeddings.py"

if [ $? -eq 0 ]; then
    echo "✅ Embeddings computation completed successfully"
else
    echo "❌ Embeddings computation failed"
    exit 1
fi

echo ""
echo "🎯 Step 4: Model Training"
echo "========================="
python "$SRC_DIR/ml/train.py"

if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully"
else
    echo "❌ Model training failed"
    exit 1
fi

echo ""
echo "📈 Step 5: Model Evaluation"
echo "==========================="
python "$SRC_DIR/ml/evaluate.py"

if [ $? -eq 0 ]; then
    echo "✅ Model evaluation completed successfully"
else
    echo "❌ Model evaluation failed"
    exit 1
fi

echo ""
echo "🎉 Pipeline Completed Successfully!"
echo "=================================="
echo "Model saved to: ${MODEL_DIR:-models}"
echo ""
echo "To start the API server:"
echo "python $SRC_DIR/api/server.py"
echo ""
echo "Then visit http://localhost:${API_PORT:-8000}/docs for API documentation"
