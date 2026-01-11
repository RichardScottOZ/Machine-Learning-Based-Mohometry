#!/usr/bin/env python3
"""
Spatial-Temporal Evolution Visualization Script

This script visualizes the spatial and temporal evolution of paleo-crustal thickness.
It generates scatter plots showing relationships between geographic coordinates, 
geologic age, and median crustal thickness.

Usage:
    python visualize_spatial_temporal.py --input <input_csv> --output <output_dir> [options]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize spatial-temporal evolution of paleo-crustal thickness'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with crustal thickness data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for generated plots (default: output)'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['basic', 'median', 'kernel', 'spatial', 'all'],
        default='all',
        help='Type of plot to generate (default: all)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output plots (default: 300)'
    )
    return parser.parse_args()


def plot_basic_correlation(df, output_path, dpi=300):
    """Plot basic correlation between crustal thickness and age."""
    print("Generating basic correlation plot...")
    
    age = df['Age']
    crustal_thickness = df['Predicted_Crustal_Thickness']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(age, crustal_thickness, color='blue', alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Crustal Thickness')
    plt.title('Correlation Between Crustal Thickness and Age')
    
    num_samples = len(df)
    plt.text(0.95, 0.95, f'Number of Samples: {num_samples}',
             transform=plt.gca().transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    plt.grid(True)
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_median_with_error(df, output_path, dpi=300):
    """Plot median crustal thickness with error bars."""
    print("Generating median correlation plot with error bars...")
    
    # Group by Age, Latitude, and Longitude and calculate median and std
    grouped = df.groupby(['Age', 'Lat', 'Lon'])['Predicted_Crustal_Thickness'].agg(['median', 'std']).reset_index()
    grouped.rename(columns={'median': 'Median_Crustal_Thickness', 'std': 'Error'}, inplace=True)
    
    df = df.merge(grouped, on=['Age', 'Lat', 'Lon'], how='left')
    
    # Group by Age to compute median and error bars
    grouped_age = df.groupby('Age').agg({
        'Median_Crustal_Thickness': 'median',
        'Error': 'mean',
        'Age error': 'mean'
    }).reset_index()
    grouped_age.rename(columns={
        'Median_Crustal_Thickness': 'Crustal_Thickness_Median',
        'Error': 'Crustal_Thickness_Error',
        'Age error': 'Age_Error'
    }, inplace=True)
    
    age = grouped_age['Age']
    crustal_thickness_median = grouped_age['Crustal_Thickness_Median']
    crustal_thickness_error = grouped_age['Crustal_Thickness_Error']
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(age, crustal_thickness_median, yerr=crustal_thickness_error,
                 fmt='o', color='blue', alpha=0.7, ecolor='blue',
                 elinewidth=1.0, capsize=5)
    plt.xlabel('Age')
    plt.ylabel('Crustal Thickness (Median)')
    plt.title('Correlation Between Crustal Thickness (Median) and Age with Error Bars')
    
    num_samples = len(df)
    plt.text(0.95, 0.95, f'Number of Samples: {num_samples}',
             transform=plt.gca().transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    plt.grid(True)
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_kernel_ridge_regression(df, output_path, dpi=300):
    """Plot kernel ridge regression trend with confidence intervals."""
    print("Generating kernel ridge regression plot...")
    
    # Try to use Times New Roman, fall back to default if not available
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except Exception:
        pass  # Use default font if Times New Roman is not available
    
    # Group by Age, Latitude, and Longitude
    grouped = df.groupby(['Age', 'Lat', 'Lon'])['Predicted_Crustal_Thickness'].agg(['median', 'std']).reset_index()
    grouped.rename(columns={'median': 'Median_Crustal_Thickness', 'std': 'Error'}, inplace=True)
    
    df = df.merge(grouped, on=['Age', 'Lat', 'Lon'], how='left')
    
    grouped_age = df.groupby('Age').agg({
        'Median_Crustal_Thickness': 'median',
        'Error': 'mean',
        'Age error': 'mean'
    }).reset_index()
    grouped_age.rename(columns={
        'Median_Crustal_Thickness': 'Crustal_Thickness_Median',
        'Error': 'Crustal_Thickness_Error',
        'Age error': 'Age_Error'
    }, inplace=True)
    
    age = grouped_age['Age']
    crustal_thickness_median = grouped_age['Crustal_Thickness_Median']
    
    # Kernel Ridge Regression
    bandwidth = 20.0
    gamma = 1 / (2 * bandwidth ** 2)
    kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=gamma)
    
    age_reshaped = age.values.reshape(-1, 1)
    kr.fit(age_reshaped, crustal_thickness_median)
    
    min_age = age.min()
    age_range = np.linspace(min_age, 250, 1000).reshape(-1, 1)
    predicted_thickness = kr.predict(age_range)
    
    # Bootstrap resampling for confidence intervals
    bootstrap_samples = 10000
    bootstrap_predictions = np.zeros((bootstrap_samples, age_range.shape[0]))
    
    print("Computing bootstrap confidence intervals...")
    for i in range(bootstrap_samples):
        if i % 1000 == 0:
            print(f"  Bootstrap iteration {i}/{bootstrap_samples}")
        resample_idx = np.random.choice(len(age), len(age), replace=True)
        kr.fit(age_reshaped[resample_idx], crustal_thickness_median.values[resample_idx])
        bootstrap_predictions[i, :] = kr.predict(age_range)
    
    ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(age, crustal_thickness_median, color='blue', alpha=0.5,
                edgecolors='black', linewidths=0.25, label='Data points')
    plt.plot(age_range, predicted_thickness, color='red',
             label='Trend (Kernel Ridge Regression)')
    plt.fill_between(age_range.flatten(), ci_lower, ci_upper,
                     color='red', alpha=0.3, label='95% Confidence Interval')
    
    plt.xlabel('Age (Ma)')
    plt.ylabel('Crustal Thickness (km)')
    plt.title('Correlation Between Crustal Thickness (Median) and Age with Kernel Ridge Regression Trend')
    plt.xlim(0, 250)
    plt.ylim(20, 90)
    
    num_samples = len(df)
    plt.text(0.95, 0.95, f'Number of Samples: {num_samples}',
             transform=plt.gca().transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_spatial_correlation(df, output_path, dpi=300):
    """Plot spatial correlation between age, lat/lon, and crustal thickness."""
    print("Generating spatial correlation plots...")
    
    # Try to use Times New Roman, fall back to default if not available
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except Exception:
        pass  # Use default font if Times New Roman is not available
    
    # Compute median and std for each combination
    grouped = df.groupby(['Age', 'Lat', 'Lon'])['Predicted_Crustal_Thickness'].agg(['median', 'std']).reset_index()
    grouped.rename(columns={'median': 'Median_Crustal_Thickness', 'std': 'Error'}, inplace=True)
    
    df = df.merge(grouped, on=['Age', 'Lat', 'Lon'], how='left')
    
    # Define axis limits
    x_limit = (0, 250)
    y_limit_lon = (108, 123)
    y_limit_lat = (19, 35)
    
    cmap = plt.get_cmap('coolwarm')
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot error bars first
    for ax, y_col in zip(axs, ['Lon', 'Lat']):
        for (age, lat, lon), group in df.groupby(['Age', 'Lat', 'Lon']):
            avg_y = group[y_col].mean()
            color = cmap((group['Median_Crustal_Thickness'].mean() - df['Predicted_Crustal_Thickness'].min()) / 
                        (df['Predicted_Crustal_Thickness'].max() - df['Predicted_Crustal_Thickness'].min()))
            ax.errorbar(age, avg_y, xerr=group['Age error'].mean(),
                       fmt='none', ecolor=color, elinewidth=0.5, capsize=3)
    
    # Upper plot: Longitude vs Age
    scatter1 = axs[0].scatter(df['Age'], df['Lon'], c=df['Median_Crustal_Thickness'],
                              cmap=cmap, alpha=0.7, edgecolor='black', linewidth=0.25)
    axs[0].set_ylabel('Longitude')
    axs[0].set_title('Correlation Between Age, Longitude, and Median Crustal Thickness')
    axs[0].set_xlim(x_limit)
    axs[0].set_ylim(y_limit_lon)
    axs[0].grid(True)
    axs[0].xaxis.set_minor_locator(MultipleLocator(10))
    axs[0].yaxis.set_minor_locator(MultipleLocator(1))
    
    # Lower plot: Latitude vs Age
    scatter2 = axs[1].scatter(df['Age'], df['Lat'], c=df['Median_Crustal_Thickness'],
                              cmap=cmap, alpha=0.7, edgecolor='black', linewidth=0.25)
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Latitude')
    axs[1].set_title('Correlation Between Age, Latitude, and Median Crustal Thickness')
    axs[1].set_xlim(x_limit)
    axs[1].set_ylim(y_limit_lat)
    axs[1].grid(True)
    axs[1].xaxis.set_minor_locator(MultipleLocator(10))
    axs[1].yaxis.set_minor_locator(MultipleLocator(1))
    
    # Create color bar
    cbar_ax = fig.add_axes([0.835, 0.125, 0.015, 0.7])
    cbar = plt.colorbar(scatter1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Median Crustal Thickness')
    
    plt.subplots_adjust(right=0.82)
    
    # Show number of samples
    num_samples = len(df)
    axs[0].text(0.95, 0.95, f'Number of Samples: {num_samples}',
                transform=axs[0].transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    axs[1].text(0.95, 0.95, f'Number of Samples: {num_samples}',
                transform=axs[1].transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['Age', 'Predicted_Crustal_Thickness', 'Lat', 'Lon', 'Age error']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} samples")
    
    # Generate plots based on type
    if args.plot_type in ['basic', 'all']:
        plot_basic_correlation(
            df,
            os.path.join(args.output, 'basic_correlation.pdf'),
            dpi=args.dpi
        )
    
    if args.plot_type in ['median', 'all']:
        plot_median_with_error(
            df,
            os.path.join(args.output, 'median_with_error.pdf'),
            dpi=args.dpi
        )
    
    if args.plot_type in ['kernel', 'all']:
        plot_kernel_ridge_regression(
            df,
            os.path.join(args.output, 'kernel_ridge_regression.pdf'),
            dpi=args.dpi
        )
    
    if args.plot_type in ['spatial', 'all']:
        plot_spatial_correlation(
            df,
            os.path.join(args.output, 'spatial_correlation.pdf'),
            dpi=args.dpi
        )
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
