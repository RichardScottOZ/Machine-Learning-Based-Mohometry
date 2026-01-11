#!/bin/bash
set -e

# Default to help if no command provided
COMMAND=${1:-help}

case "$COMMAND" in
  train)
    shift
    python /app/src/predict_crustal_thickness.py --mode train "$@"
    ;;
  predict)
    shift
    python /app/src/predict_crustal_thickness.py --mode predict "$@"
    ;;
  visualize)
    shift
    python /app/src/visualize_spatial_temporal.py "$@"
    ;;
  help)
    echo "Machine Learning Based Mohometry - Container Entrypoint"
    echo ""
    echo "Usage: docker run [OPTIONS] <image> <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  train       Train a new crustal thickness prediction model"
    echo "  predict     Use a trained model to predict crustal thickness"
    echo "  visualize   Generate spatial-temporal visualization plots"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Train a model"
    echo "  docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output <image> train \\"
    echo "    --train-data /data/Model_1.csv --output /output/models"
    echo ""
    echo "  # Make predictions"
    echo "  docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output <image> predict \\"
    echo "    --model /output/models/model.pkl --scaler /output/models/scaler.pkl \\"
    echo "    --input /data/new_data.csv --output /output/predictions.csv"
    echo ""
    echo "  # Generate visualizations"
    echo "  docker run -v \$(pwd)/data:/data -v \$(pwd)/output:/output <image> visualize \\"
    echo "    --input /data/spatial_data.csv --output /output/plots"
    echo ""
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Run 'docker run <image> help' for usage information"
    exit 1
    ;;
esac
