"""
Evaluation Script for Impossible Travel Detector

This script evaluates the trained model on the test set and generates:
- Performance metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Classification report
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ImpossibleTravelDetector


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(config):
    """
    Load test data from splits.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_test, y_test)
    """
    test_df = pd.read_csv(config['data']['test_split_path'])
    X_test = test_df.drop(columns=['is_impossible_travel'])
    y_test = test_df['is_impossible_travel']
    
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Impossible Travel'],
        yticklabels=['Normal', 'Impossible Travel']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, save_path='precision_recall_curve.png'):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {save_path}")
    plt.close()


def plot_threshold_analysis(y_true, y_proba, save_path='threshold_analysis.png'):
    """
    Plot how metrics change with different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        
        if y_pred.sum() == 0:  # No positive predictions
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
        else:
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Threshold analysis saved to: {save_path}")
    plt.close()


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_proba),
        'average_precision': average_precision_score(y_true, y_proba)
    }
    
    return metrics


def print_evaluation_report(metrics, y_true, y_pred):
    """
    Print comprehensive evaluation report.
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nPerformance Metrics:")
    print("-"*60)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    print(f"  Average Precision:  {metrics['average_precision']:.4f}")
    
    print("\nConfusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_true, y_pred)
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    print("\nDetailed Classification Report:")
    print("-"*60)
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Impossible Travel'],
        digits=4
    ))
    
    print("="*60)


def save_evaluation_results(metrics, y_true, y_pred, save_dir):
    """
    Save evaluation results to files.
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
        save_dir: Directory to save results
    """
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save classification report
    report = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Impossible Travel'],
        digits=4
    )
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Normal', 'Actual Impossible Travel'],
        columns=['Predicted Normal', 'Predicted Impossible Travel']
    )
    cm_path = os.path.join(save_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


def main(config_path, model_path=None, threshold=None):
    """
    Main evaluation pipeline.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model (optional, uses latest if not provided)
        threshold: Classification threshold (optional, uses config default)
    """
    print("\n" + "="*60)
    print("IMPOSSIBLE TRAVEL DETECTOR - EVALUATION")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)
    
    # Set threshold
    if threshold is None:
        threshold = config['evaluation']['threshold']
    print(f"Using classification threshold: {threshold}")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data(config)
    print(f"Test set size: {len(X_test)}")
    print(f"Positive samples: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Load model
    print("\nLoading trained model...")
    if model_path is None:
        # Find latest model in save directory
        save_dir = config['model_save']['save_dir']
        models = [f for f in os.listdir(save_dir) if f.endswith('.keras')]
        if not models:
            raise ValueError(f"No trained models found in {save_dir}")
        model_path = os.path.join(save_dir, sorted(models)[-1])
    
    input_dim = X_test.shape[1]
    detector = ImpossibleTravelDetector(config, input_dim)
    detector.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Make predictions
    print("\nGenerating predictions...")
    y_proba = detector.predict_proba(X_test.values).flatten()
    y_pred = (y_proba > threshold).astype(int)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_test.values, y_pred, y_proba)
    
    # Print evaluation report
    print_evaluation_report(metrics, y_test.values, y_pred)
    
    # Create output directory
    output_dir = os.path.join(config['model_save']['save_dir'], 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    print("\nSaving evaluation results...")
    save_evaluation_results(metrics, y_test.values, y_pred, output_dir)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        y_test.values, y_pred,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    plot_roc_curve(
        y_test.values, y_proba,
        save_path=os.path.join(output_dir, 'roc_curve.png')
    )
    plot_precision_recall_curve(
        y_test.values, y_proba,
        save_path=os.path.join(output_dir, 'precision_recall_curve.png')
    )
    plot_threshold_analysis(
        y_test.values, y_proba,
        save_path=os.path.join(output_dir, 'threshold_analysis.png')
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - evaluation_metrics.csv")
    print("  - classification_report.txt")
    print("  - confusion_matrix.csv")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - threshold_analysis.png")
    print("\n" + "="*60 + "\n")
    
    return metrics, y_pred, y_proba


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Impossible Travel Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (optional, uses latest if not provided)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Classification threshold (optional, uses config default)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        metrics, y_pred, y_proba = main(args.config, args.model, args.threshold)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
