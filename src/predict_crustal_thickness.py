#!/usr/bin/env python3
"""
Crustal Thickness Prediction Script

This script uses machine learning (CatBoost) to predict paleo-crustal thickness
from geochemical element data. It supports both training and prediction modes.

Usage:
    # Training mode
    python predict_crustal_thickness.py --mode train --train-data <training_csv> --output <output_dir>
    
    # Prediction mode
    python predict_crustal_thickness.py --mode predict --model <model_path> --input <input_csv> --output <output_csv>
"""

import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train or use CatBoost model for crustal thickness prediction'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        required=True,
        help='Operation mode: train a new model or use existing model for prediction'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        help='Path to training data CSV (required for train mode)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input data CSV for prediction (required for predict mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path (directory for train mode, CSV file for predict mode)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to saved model (required for predict mode)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        help='Path to saved scaler (required for predict mode)'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=8,
        help='CatBoost depth parameter (default: 8)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1400,
        help='CatBoost iterations parameter (default: 1400)'
    )
    parser.add_argument(
        '--l2-leaf-reg',
        type=int,
        default=10,
        help='CatBoost L2 regularization parameter (default: 10)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='CatBoost learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=10,
        help='Number of cross-validation folds for training (default: 10)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output plots (default: 300)'
    )
    return parser.parse_args()


def check_nan_values(df):
    """Check for NaN values in DataFrame."""
    if df.isnull().values.any():
        print("WARNING: The CSV file contains NaN values.")
        columns_with_nan = df.columns[df.isnull().any()].tolist()
        print(f"Columns with NaN values: {columns_with_nan}")
        rows_with_nan = df[df.isnull().any(axis=1)]
        print(f"Number of rows with NaN values: {len(rows_with_nan)}")
        return True
    else:
        print("No NaN values found in the dataset.")
        return False


def train_model(args):
    """Train a new CatBoost model."""
    print(f"Loading training data from {args.train_data}...")
    
    # Load training data
    try:
        data = pd.read_csv(args.train_data)
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} training samples")
    
    # Check for NaN values
    check_nan_values(data)
    
    # Extract features and target
    # Assuming the last column is 'Crustal_Thickness' and all others are features
    if 'Crustal_Thickness' not in data.columns:
        print("Error: 'Crustal_Thickness' column not found in training data")
        sys.exit(1)
    
    # Get all columns except Crustal_Thickness as features
    feature_cols = [col for col in data.columns if col != 'Crustal_Thickness']
    x = data[feature_cols].values
    y = data['Crustal_Thickness'].values
    
    print(f"Number of features: {x.shape[1]}")
    print(f"Target variable: Crustal_Thickness")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Set up CatBoost parameters
    best_params = {
        'depth': args.depth,
        'iterations': args.iterations,
        'l2_leaf_reg': args.l2_leaf_reg,
        'learning_rate': args.learning_rate
    }
    
    print(f"Training CatBoost model with parameters: {best_params}")
    
    # Initialize model
    best_regr = CatBoostRegressor(**best_params, verbose=0)
    
    # Perform cross-validation
    print(f"Performing {args.cv_folds}-fold cross-validation...")
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    y_predict = np.zeros_like(y)
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x_scaled)):
        print(f"  Training fold {fold_idx + 1}/{args.cv_folds}...")
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create a new model instance for each fold to ensure clean state
        fold_regr = CatBoostRegressor(**best_params, verbose=0)
        fold_regr.fit(x_train, y_train.ravel(),
                     eval_set=(x_test, y_test),
                     early_stopping_rounds=50)
        y_predict[test_index] = fold_regr.predict(x_test)
    
    # Evaluate the model
    r2_test = r2_score(y, y_predict)
    rmse = mean_squared_error(y, y_predict, squared=False)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2_test:.3f}")
    print(f"  RMSE: {rmse:.1f}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    best_regr = CatBoostRegressor(**best_params, verbose=0)
    best_regr.fit(x_scaled, y.ravel())
    
    # Save model and scaler
    model_path = os.path.join(args.output, 'model.pkl')
    scaler_path = os.path.join(args.output, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_regr, f)
    print(f"Model saved to {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join(args.output, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for name in feature_cols:
            f.write(f"{name}\n")
    print(f"Feature names saved to {feature_names_path}")
    
    # Plot results
    print("\nGenerating prediction plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_predict, 25, color='blue', edgecolors='black',
               alpha=0.2, linewidths=0.25)
    ax.plot([0, 90], [0, 90], linestyle='--', lw=1, color='b', alpha=.8)
    ax.plot([10, 90], [0, 80], linestyle='--', lw=1, color='g', alpha=.5)
    ax.plot([0, 80], [10, 90], linestyle='--', lw=1, color='g', alpha=.5)
    ax.text(10, 75, r'$R^2 = {:.3f}$'.format(r2_test), fontsize=15)
    ax.text(10, 70, r'RMSE = {:.1f}'.format(rmse), fontsize=15)
    ax.set_title('Crustal Thickness')
    ax.set_xlabel('Observed', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.axis([0, 90, 0, 90])
    
    plot_path = os.path.join(args.output, 'prediction_plot.pdf')
    plt.savefig(plot_path, format='pdf', dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to {plot_path}")
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    feature_importances = best_regr.get_feature_importance()
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    })
    
    sorted_feature_importance_df = feature_importance_df.sort_values(
        by='Importance', ascending=True
    )
    
    # Save feature importance to CSV
    importance_csv_path = os.path.join(args.output, 'feature_importance.csv')
    sorted_feature_importance_df.to_csv(importance_csv_path, index=False)
    print(f"Feature importance saved to {importance_csv_path}")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(sorted_feature_importance_df['Feature'],
            sorted_feature_importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    
    mean_importance = sorted_feature_importance_df['Importance'].mean()
    ax.axvline(x=mean_importance, color='r', linestyle='--',
               label='Mean Importance')
    ax.legend()
    
    importance_plot_path = os.path.join(args.output, 'feature_importance.pdf')
    plt.savefig(importance_plot_path, format='pdf', dpi=args.dpi,
                bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {importance_plot_path}")
    
    print("\nTraining complete!")


def predict_with_model(args):
    """Use trained model to make predictions."""
    print(f"Loading model from {args.model}...")
    
    # Load model
    try:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load scaler
    print(f"Loading scaler from {args.scaler}...")
    try:
        with open(args.scaler, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        sys.exit(1)
    
    # Load input data
    print(f"Loading input data from {args.input}...")
    try:
        new_data = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading input data: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(new_data)} samples for prediction")
    
    # Check for NaN values
    check_nan_values(new_data)
    
    # Standardize features
    print("Standardizing features...")
    x_scaled_new = scaler.transform(new_data.values)
    
    # Predict
    print("Making predictions...")
    predicted_thickness_new = model.predict(x_scaled_new)
    
    # Add predictions to dataframe
    new_data['Predicted_Crustal_Thickness'] = predicted_thickness_new
    
    # Save output
    print(f"Saving predictions to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    new_data.to_csv(args.output, index=False)
    
    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {predicted_thickness_new.mean():.2f}")
    print(f"  Std: {predicted_thickness_new.std():.2f}")
    print(f"  Min: {predicted_thickness_new.min():.2f}")
    print(f"  Max: {predicted_thickness_new.max():.2f}")
    
    print("\nPrediction complete!")


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.mode == 'train':
        if not args.train_data:
            print("Error: --train-data is required for train mode")
            sys.exit(1)
        train_model(args)
    elif args.mode == 'predict':
        if not args.model or not args.scaler or not args.input:
            print("Error: --model, --scaler, and --input are required for predict mode")
            sys.exit(1)
        predict_with_model(args)


if __name__ == '__main__':
    main()
