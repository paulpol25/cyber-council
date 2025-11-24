"""
Training Script for Impossible Travel Detector

This script orchestrates the complete training pipeline:
1. Load and preprocess data
2. Engineer features
3. Build and train the model
4. Save the trained model
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import ImpossibleTravelDetector


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history metrics.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def print_data_summary(data):
    """
    Print summary statistics of the dataset.
    
    Args:
        data: Dictionary containing data splits
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nTraining set:")
    print(f"  Features shape: {data['X_train'].shape}")
    print(f"  Labels shape: {data['y_train'].shape}")
    print(f"  Positive samples: {data['y_train'].sum()} ({data['y_train'].mean()*100:.2f}%)")
    
    print(f"\nValidation set:")
    print(f"  Features shape: {data['X_val'].shape}")
    print(f"  Labels shape: {data['y_val'].shape}")
    print(f"  Positive samples: {data['y_val'].sum()} ({data['y_val'].mean()*100:.2f}%)")
    
    print(f"\nTest set:")
    print(f"  Features shape: {data['X_test'].shape}")
    print(f"  Labels shape: {data['y_test'].shape}")
    print(f"  Positive samples: {data['y_test'].sum()} ({data['y_test'].mean()*100:.2f}%)")
    
    print(f"\nTotal features: {data['X_train'].shape[1]}")
    print("\n" + "="*60)


def main(config_path, skip_preprocessing=False):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
        skip_preprocessing: If True, load existing preprocessed data
    """
    print("\n" + "="*60)
    print("IMPOSSIBLE TRAVEL DETECTOR - TRAINING PIPELINE")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)
    
    # Step 1: Data Preprocessing
    print("\n" + "-"*60)
    print("STEP 1: DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("-"*60)
    
    if not skip_preprocessing:
        # Load raw data
        raw_data_path = config['data']['raw_data_path']
        df = pd.read_csv(raw_data_path)
        
        print(f"\nLoaded raw data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Feature Engineering
        print("\nApplying feature engineering...")
        feature_engineer = FeatureEngineer(config)
        df = feature_engineer.engineer_features(df)
        
        # Save engineered dataset temporarily
        temp_engineered_path = 'data/processed/engineered_data.csv'
        os.makedirs(os.path.dirname(temp_engineered_path), exist_ok=True)
        df.to_csv(temp_engineered_path, index=False)
        
        # Data Preprocessing
        print("\nPreprocessing data...")
        preprocessor = DataPreprocessor(config)
        data = preprocessor.preprocess(temp_engineered_path, save_splits=True)
        
    else:
        print("\nLoading preprocessed data from splits...")
        data = {
            'X_train': pd.read_csv(config['data']['train_split_path']).drop(columns=['is_impossible_travel']),
            'y_train': pd.read_csv(config['data']['train_split_path'])['is_impossible_travel'],
            'X_val': pd.read_csv(config['data']['val_split_path']).drop(columns=['is_impossible_travel']),
            'y_val': pd.read_csv(config['data']['val_split_path'])['is_impossible_travel'],
            'X_test': pd.read_csv(config['data']['test_split_path']).drop(columns=['is_impossible_travel']),
            'y_test': pd.read_csv(config['data']['test_split_path'])['is_impossible_travel']
        }
    
    # Print data summary
    print_data_summary(data)
    
    # Step 2: Model Building and Training
    print("\n" + "-"*60)
    print("STEP 2: MODEL BUILDING AND TRAINING")
    print("-"*60)
    
    # Get input dimension
    input_dim = data['X_train'].shape[1]
    
    # Initialize model
    print(f"\nInitializing model with {input_dim} input features...")
    detector = ImpossibleTravelDetector(config, input_dim)
    
    # Build model
    detector.build_model()
    detector.summary()
    
    # Train model
    print("\n" + "-"*60)
    print("Starting training...")
    print("-"*60)
    
    history = detector.train(
        data['X_train'].values,
        data['y_train'].values,
        data['X_val'].values,
        data['y_val'].values
    )
    
    # Step 3: Save Model and Results
    print("\n" + "-"*60)
    print("STEP 3: SAVING MODEL AND RESULTS")
    print("-"*60)
    
    # Save final model
    model_path = detector.save_model()
    
    # Plot training history
    plot_path = os.path.join(config['model_save']['save_dir'], 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Print final metrics
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    final_epoch = len(history.history['loss'])
    print(f"\nTotal epochs trained: {final_epoch}")
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Precision: {history.history['precision'][-1]:.4f}")
    print(f"  Recall: {history.history['recall'][-1]:.4f}")
    print(f"  AUC: {history.history['auc'][-1]:.4f}")
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Precision: {history.history['val_precision'][-1]:.4f}")
    print(f"  Recall: {history.history['val_recall'][-1]:.4f}")
    print(f"  AUC: {history.history['val_auc'][-1]:.4f}")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Training history plot saved to: {plot_path}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run evaluate.py to assess model performance on test set")
    print("2. Check training_history.png for learning curves")
    print("3. Review model checkpoints in models/checkpoints/")
    print("="*60 + "\n")
    
    return detector, history, data


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Impossible Travel Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing and load existing data splits'
    )
    
    args = parser.parse_args()
    
    # Run training
    try:
        detector, history, data = main(args.config, args.skip_preprocessing)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
