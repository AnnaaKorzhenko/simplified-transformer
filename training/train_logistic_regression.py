"""
Train Logistic Regression model on temporal logic datasets.
Supports both small (100 sequences) and large (10,000 sequences) datasets.
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from typing import List, Dict
import matplotlib.pyplot as plt

from training.utils import extract_features, split_dataset, split_dataset_three_way, load_single_dataset


def train_logistic_regression(X_train: List[List[str]], y_train: List[int],
                              X_test: List[List[str]], y_test: List[int],
                              alphabet: List[str], sequence_length: int,
                              X_val: List[List[str]] = None,
                              y_val: List[int] = None,
                              random_state: int = 42) -> Dict:
    """Train logistic regression model."""
    # Extract features
    X_train_features = np.array([extract_features(seq, alphabet, sequence_length) for seq in X_train])
    X_test_features = np.array([extract_features(seq, alphabet, sequence_length) for seq in X_test])
    
    # Train
    model = LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear')
    model.fit(X_train_features, y_train)
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        X_val_features = np.array([extract_features(seq, alphabet, sequence_length) for seq in X_val])
        y_val_pred = model.predict(X_val_features)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Predict on test set
    y_pred = model.predict(X_test_features)
    y_pred_proba = model.predict_proba(X_test_features)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }


def train_on_dataset(dataset_dir: str, formula_id: int = 33, output_dir: str = None):
    """Train logistic regression on a dataset and save results."""
    print("=" * 80)
    print(f"Training Logistic Regression")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Formula ID: {formula_id}\n")
    
    # Load dataset
    print("Loading dataset...")
    formula, dataset, alphabet, sequence_length = load_single_dataset(dataset_dir, formula_id)
    
    print(f"Loaded dataset: {len(dataset)} sequences")
    print(f"Alphabet size: {len(alphabet)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Positive: {sum(1 for _, label in dataset if label == 1)}")
    print(f"Negative: {sum(1 for _, label in dataset if label == 0)}\n")
    
    # Split dataset into train/val/test
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    )
    print(f"Train set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences\n")
    
    # Train
    print("Training Logistic Regression...")
    result = train_logistic_regression(
        X_train, y_train, X_test, y_test, alphabet, sequence_length,
        X_val=X_val, y_val=y_val, random_state=42
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1 Score:  {result['f1']:.4f}")
    print(f"  AUC-ROC:   {result['auc']:.4f}")
    print(f"  Confusion Matrix:\n    {result['confusion_matrix']}\n")
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={result['auc']:.3f})", linewidth=2.5, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Logistic Regression ({dataset_dir})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save results
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = project_root
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ROC curve
    dataset_name = os.path.basename(dataset_dir.rstrip('/'))
    roc_path = os.path.join(output_dir, f'roc_curve_logistic_regression_{dataset_name}.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to: {roc_path}")
    plt.close()
    
    # Save results JSON
    output = {
        'dataset_info': {
            'dataset': dataset_dir,
            'formula_id': formula_id,
            'alphabet_size': len(alphabet),
            'sequence_length': sequence_length,
            'total_sequences': len(dataset),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'results': {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'auc': float(result['auc']),
            'confusion_matrix': result['confusion_matrix']
        }
    }
    
    results_path = os.path.join(output_dir, f'logistic_regression_results_{dataset_name}.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved results to: {results_path}")
    print(f"{'='*80}\n")
    
    return result


def main():
    """Main function to train on both small and large datasets."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Train on small dataset
    print("\n" + "="*80)
    print("TRAINING ON SMALL DATASET (100 sequences)")
    print("="*80 + "\n")
    train_on_dataset("single_dataset", formula_id=33, output_dir=project_root)
    
    # Train on large dataset
    print("\n" + "="*80)
    print("TRAINING ON LARGE DATASET (10,000 sequences)")
    print("="*80 + "\n")
    train_on_dataset("large_single_dataset", formula_id=33, output_dir=project_root)


if __name__ == "__main__":
    main()



