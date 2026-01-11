# Machine Learning Based Mohometry - AWS Batch / Container Edition

This document describes how to use the containerized version of the Machine Learning Based Mohometry tools for AWS Batch or other container orchestration platforms.

## Overview

The repository now includes a `src/` directory with Python scripts that can be executed in containers, making it suitable for cloud environments like AWS Batch, Kubernetes, or any Docker-based workflow.

## Contents

- `src/predict_crustal_thickness.py` - Train models and predict crustal thickness
- `src/visualize_spatial_temporal.py` - Generate spatial-temporal visualization plots
- `Dockerfile` - Container image definition
- `entrypoint.sh` - Container entrypoint script
- `requirements.txt` - Python dependencies

## Quick Start

### Building the Docker Image

```bash
docker build -t mohometry:latest .
```

### Running the Container

#### 1. Train a Model

```bash
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output mohometry:latest train \
  --train-data /data/Models\ for\ machine\ learning\ data\ training/Model\ 1.csv \
  --output /output/models
```

This will:
- Load training data from the mounted data directory
- Train a CatBoost model with cross-validation
- Save the model, scaler, and feature importance plots to the output directory

#### 2. Make Predictions

```bash
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output mohometry:latest predict \
  --model /output/models/model.pkl \
  --scaler /output/models/scaler.pkl \
  --input /data/Paleo/South\ China\ Block/Model1_imputed_South\ China.csv \
  --output /output/predictions.csv
```

This will:
- Load the trained model and scaler
- Predict crustal thickness for new data
- Save predictions to a CSV file

#### 3. Generate Visualizations

```bash
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output mohometry:latest visualize \
  --input /data/Training\ data\ with\ locations\ and\ citations/Table\ S3.csv \
  --output /output/plots
```

This will:
- Generate various spatial-temporal correlation plots
- Save plots as PDF files

## Script Usage

### predict_crustal_thickness.py

#### Training Mode

```bash
python src/predict_crustal_thickness.py --mode train \
  --train-data <path_to_training_csv> \
  --output <output_directory> \
  [--depth 8] \
  [--iterations 1400] \
  [--l2-leaf-reg 10] \
  [--learning-rate 0.1] \
  [--cv-folds 10]
```

**Parameters:**
- `--mode train` - Training mode
- `--train-data` - Path to training CSV file (must contain 'Crustal_Thickness' column)
- `--output` - Output directory for model, scaler, and plots
- `--depth` - CatBoost depth parameter (default: 8)
- `--iterations` - CatBoost iterations (default: 1400)
- `--l2-leaf-reg` - L2 regularization (default: 10)
- `--learning-rate` - Learning rate (default: 0.1)
- `--cv-folds` - Number of cross-validation folds (default: 10)

**Outputs:**
- `model.pkl` - Trained CatBoost model
- `scaler.pkl` - StandardScaler for feature normalization
- `feature_names.txt` - List of feature names
- `prediction_plot.pdf` - Model performance plot
- `feature_importance.pdf` - Feature importance plot
- `feature_importance.csv` - Feature importance values

#### Prediction Mode

```bash
python src/predict_crustal_thickness.py --mode predict \
  --model <path_to_model.pkl> \
  --scaler <path_to_scaler.pkl> \
  --input <path_to_input_csv> \
  --output <path_to_output_csv>
```

**Parameters:**
- `--mode predict` - Prediction mode
- `--model` - Path to trained model file
- `--scaler` - Path to scaler file
- `--input` - Path to input CSV file with geochemical data
- `--output` - Path to output CSV file with predictions

### visualize_spatial_temporal.py

```bash
python src/visualize_spatial_temporal.py \
  --input <path_to_input_csv> \
  --output <output_directory> \
  [--plot-type {basic,median,kernel,spatial,all}] \
  [--dpi 300]
```

**Parameters:**
- `--input` - Path to input CSV with columns: Age, Predicted_Crustal_Thickness, Lat, Lon, Age error
- `--output` - Output directory for plots
- `--plot-type` - Type of plot to generate (default: all)
  - `basic` - Basic correlation between thickness and age
  - `median` - Median correlation with error bars
  - `kernel` - Kernel ridge regression with confidence intervals
  - `spatial` - Spatial correlation plots (lat/lon vs age)
  - `all` - Generate all plot types
- `--dpi` - DPI for output plots (default: 300)

**Outputs:**
- `basic_correlation.pdf` - Basic scatter plot
- `median_with_error.pdf` - Median values with error bars
- `kernel_ridge_regression.pdf` - Trend analysis with confidence intervals
- `spatial_correlation.pdf` - Spatial-temporal correlation

## AWS Batch Setup

### 1. Push Image to ECR

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag mohometry:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/mohometry:latest

# Push image
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/mohometry:latest
```

### 2. Create Job Definition

Create an AWS Batch job definition with:
- **Image**: Your ECR image URI
- **vCPUs**: 2-4 (depending on data size)
- **Memory**: 4096-8192 MB (depending on data size)
- **Job Role**: IAM role with S3 access for data

### 3. Submit Job

Example job submission for training:

```json
{
  "jobName": "mohometry-train",
  "jobQueue": "your-job-queue",
  "jobDefinition": "mohometry-job-def",
  "containerOverrides": {
    "command": [
      "train",
      "--train-data", "/data/Model_1.csv",
      "--output", "/output/models"
    ],
    "environment": [
      {
        "name": "INPUT_S3_BUCKET",
        "value": "your-data-bucket"
      },
      {
        "name": "OUTPUT_S3_BUCKET",
        "value": "your-output-bucket"
      }
    ]
  }
}
```

### 4. Example Workflow with S3

For a complete AWS Batch workflow with S3 data:

1. **Pre-Job**: Download data from S3 to `/data`
2. **Main Job**: Run container with mounted volumes
3. **Post-Job**: Upload results from `/output` to S3

You can use an init container or AWS Batch array jobs to orchestrate this workflow.

## Local Development

### Without Docker

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run scripts directly:
```bash
python src/predict_crustal_thickness.py --mode train --train-data data/Models\ for\ machine\ learning\ data\ training/Model\ 1.csv --output output/models

python src/visualize_spatial_temporal.py --input data/Training\ data\ with\ locations\ and\ citations/Table\ S3.csv --output output/plots
```

## Data Requirements

### Training Data
- CSV format with geochemical elements as features
- Must include `Crustal_Thickness` column as target
- Example: `data/Models for machine learning data training/Model 1.csv`

### Prediction Data
- CSV format with same features as training data
- No `Crustal_Thickness` column needed
- Example: `data/Paleo/South China Block/Model1_imputed_South China.csv`

### Visualization Data
- CSV format with columns: `Age`, `Predicted_Crustal_Thickness`, `Lat`, `Lon`, `Age error`
- Example: `data/Training data with locations and citations/Table S3.csv`

## Performance Considerations

- **Training**: Can take 5-30 minutes depending on data size and CV folds
- **Prediction**: Fast, typically completes in seconds
- **Visualization**: Kernel ridge regression with bootstrap can take 5-10 minutes

For AWS Batch:
- Use appropriate instance types (e.g., c5.xlarge for compute-intensive training)
- Consider splitting large datasets into smaller batches for parallel processing
- Use S3 for data storage and results

## Troubleshooting

### Container Issues

1. **Permission errors**: Ensure mounted volumes have correct permissions
2. **Memory errors**: Increase container memory allocation
3. **Missing data**: Verify volume mounts are correct

### Script Issues

1. **NaN values**: The scripts will warn about missing values in data
2. **Missing columns**: Ensure input data has required columns
3. **File paths**: Use absolute paths inside containers (e.g., `/data/file.csv`)

## License

See LICENSE file in the repository root.

## Citation

If you use this code, please cite:

Zhou, J., Farahbakhsh, E., Williams, S., Li, X., Liu, Y., Li, S., & Müller, R. D. (2025). Machine learning and big data mining reveal Earth's deep time crustal thickness and tectonic evolution: A new chemical mohometry approach. Journal of Geophysical Research: Solid Earth, 130, e2024JB030404. https://doi.org/10.1029/2024JB030404
