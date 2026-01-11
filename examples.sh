#!/bin/bash
# Example usage script for local testing

# This script demonstrates how to use the containerized tools locally
# before deploying to AWS Batch

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Machine Learning Based Mohometry Examples"
echo "=========================================="
echo ""

# Example 1: Train a model
echo "Example 1: Train a Model"
echo "------------------------"
echo "Command:"
echo "python3 src/predict_crustal_thickness.py --mode train \\"
echo "  --train-data 'data/Models for machine learning data training/Model 1.csv' \\"
echo "  --output output/models \\"
echo "  --iterations 100 \\"
echo "  --cv-folds 3"
echo ""
echo "This will:"
echo "  - Load training data from Model 1.csv"
echo "  - Train a CatBoost model with 3-fold cross-validation"
echo "  - Save model, scaler, and plots to output/models/"
echo ""

# Example 2: Make predictions
echo "Example 2: Make Predictions"
echo "---------------------------"
echo "Command:"
echo "python3 src/predict_crustal_thickness.py --mode predict \\"
echo "  --model output/models/model.pkl \\"
echo "  --scaler output/models/scaler.pkl \\"
echo "  --input 'data/Paleo/South China Block/Model1_imputed_South China.csv' \\"
echo "  --output output/predictions.csv"
echo ""
echo "This will:"
echo "  - Load the trained model and scaler"
echo "  - Predict crustal thickness for South China Block data"
echo "  - Save predictions to output/predictions.csv"
echo ""

# Example 3: Visualize data
echo "Example 3: Generate Visualizations"
echo "-----------------------------------"
echo "Command:"
echo "python3 src/visualize_spatial_temporal.py \\"
echo "  --input 'data/Training data with locations and citations/Table S3.csv' \\"
echo "  --output output/plots \\"
echo "  --plot-type all"
echo ""
echo "This will:"
echo "  - Generate all visualization types"
echo "  - Save plots to output/plots/"
echo ""

# Docker examples
echo "=========================================="
echo "Docker Usage Examples"
echo "=========================================="
echo ""

echo "Build the Docker image:"
echo "docker build -t mohometry:latest ."
echo ""

echo "Run training in Docker:"
echo "docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output mohometry:latest train \\"
echo "  --train-data '/data/Models for machine learning data training/Model 1.csv' \\"
echo "  --output /output/models \\"
echo "  --iterations 100 \\"
echo "  --cv-folds 3"
echo ""

echo "Run prediction in Docker:"
echo "docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output mohometry:latest predict \\"
echo "  --model /output/models/model.pkl \\"
echo "  --scaler /output/models/scaler.pkl \\"
echo "  --input '/data/Paleo/South China Block/Model1_imputed_South China.csv' \\"
echo "  --output /output/predictions.csv"
echo ""

echo "Run visualization in Docker:"
echo "docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output mohometry:latest visualize \\"
echo "  --input '/data/Training data with locations and citations/Table S3.csv' \\"
echo "  --output /output/plots"
echo ""

echo "=========================================="
echo "AWS Batch Deployment Notes"
echo "=========================================="
echo ""
echo "1. Push image to ECR:"
echo "   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com"
echo "   docker tag mohometry:latest <account>.dkr.ecr.us-east-1.amazonaws.com/mohometry:latest"
echo "   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/mohometry:latest"
echo ""
echo "2. Create AWS Batch job definition using the ECR image"
echo ""
echo "3. Configure S3 for input/output data"
echo ""
echo "4. Submit jobs with appropriate commands (train/predict/visualize)"
echo ""
echo "For detailed instructions, see CONTAINER_README.md"
echo ""
