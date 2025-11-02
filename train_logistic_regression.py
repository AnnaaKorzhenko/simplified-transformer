"""
Logistic Regression models to classify sequences as satisfying or not satisfying formulas.
"""

import json
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

from ltl_formulas.formula_generator import FormulaEvaluator, load_dataset_csv, load_formula_json


def extract_features(sequence: List[str], alphabet: List[str], sequence_length: int) -> np.ndarray:
    """
    Extract features from a sequence for logistic regression.
    
    Features:
    1. Which symbol appears at each position (one-hot encoding)
    2. Frequency counts (how many times each symbol appears)
    3. First occurrence positions (normalized)
    4. Last occurrence positions (normalized)
    """
    features = []
    
    # 1. Position-based features: one-hot encoding for each position
    for pos in range(sequence_length):
        position_features = [0.0] * len(alphabet)
        if pos < len(sequence):
            if sequence[pos] in alphabet:
                idx = alphabet.index(sequence[pos])
                position_features[idx] = 1.0
        features.extend(position_features)
    
    # 2. Frequency counts: how many times each symbol appears
    symbol_counts = [sequence.count(symbol) for symbol in alphabet]
    features.extend(symbol_counts)
    
    # 3. First occurrence positions (normalized)
    first_positions = []
    for symbol in alphabet:
        try:
            first_pos = sequence.index(symbol)
        except ValueError:
            first_pos = sequence_length  # If not found, assign max position
        first_positions.append(float(first_pos) / sequence_length)  # Normalize
    features.extend(first_positions)
    
    # 4. Last occurrence positions (normalized)
    last_positions = []
    for symbol in alphabet:
        try:
            last_pos = len(sequence) - 1 - sequence[::-1].index(symbol)
        except ValueError:
            last_pos = -1  # Not found
        last_positions.append(float(max(0, last_pos)) / sequence_length)  # Normalize
    features.extend(last_positions)
    
    return np.array(features)


def load_all_datasets(dataset_dir: str = "generated_formulas_datasets") -> List[Tuple]:
    """
    Load all datasets.
    Returns list of (formula_id, formula, dataset, alphabet) tuples.
    """
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_dir)
    
    results = []
    
    formulas_dir = os.path.join(dataset_dir, "formulas")
    datasets_dir = os.path.join(dataset_dir, "datasets")
    
    use_subdirs = os.path.exists(formulas_dir) and os.path.exists(datasets_dir)
    
    # Load summary to get all formula IDs
    summary_path = os.path.join(dataset_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        for result in summary['results']:
            formula_id = result['formula_id']
            
            # Load formula
            if use_subdirs:
                formula_path = os.path.join(formulas_dir, f"formula_{formula_id}.json")
                dataset_path = os.path.join(datasets_dir, f"dataset_{formula_id}.csv")
            else:
                formula_path = os.path.join(dataset_dir, f"formula_{formula_id}.json")
                dataset_path = os.path.join(dataset_dir, f"dataset_{formula_id}.csv")
            
            if os.path.exists(formula_path) and os.path.exists(dataset_path):
                formula, metadata = load_formula_json(formula_path)
                alphabet = metadata.get('alphabet', ['p1', 'p2', 'p3', 'p4', 'p5'])
                dataset = load_dataset_csv(dataset_path)
                results.append((formula_id, formula, dataset, alphabet))
    else:
        # Or try to load formula_1 through formula_15
        for formula_id in range(1, 16):
            if use_subdirs:
                formula_path = os.path.join(formulas_dir, f"formula_{formula_id}.json")
                dataset_path = os.path.join(datasets_dir, f"dataset_{formula_id}.csv")
            else:
                formula_path = os.path.join(dataset_dir, f"formula_{formula_id}.json")
                dataset_path = os.path.join(dataset_dir, f"dataset_{formula_id}.csv")
            
            if os.path.exists(formula_path) and os.path.exists(dataset_path):
                formula, metadata = load_formula_json(formula_path)
                alphabet = metadata.get('alphabet', ['p1', 'p2', 'p3', 'p4', 'p5'])
                dataset = load_dataset_csv(dataset_path)
                results.append((formula_id, formula, dataset, alphabet))
    
    return results


def train_and_evaluate(formula_id: int, formula, dataset: List[Tuple[List[str], int]], 
                       alphabet: List[str], sequence_length: int = 10, 
                       test_size: float = 0.2, random_state: int = 42):
    """
    Train logistic regression and evaluate on a single formula.
    
    """
    # Extract features and labels
    X = np.array([extract_features(seq, alphabet, sequence_length) for seq, _ in dataset])
    y = np.array([label for _, label in dataset])
    
    # Split into train and test
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    except ValueError as e:
        print(f"Could not calculate AUC for formula {formula_id}: {e}")
        auc_score = 0.0
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([1])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'formula_id': formula_id,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds),
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def train_on_combined_dataset(all_data: List[Tuple], sequence_length: int = 10, 
                              test_size: float = 0.2, random_state: int = 42):
    """
    Train a single logistic regression model on the combined dataset from all formulas.
    """
    print("=" * 70)
    print("Combining All Formulas into Single Dataset")
    print("=" * 70)
    
    # Combine all datasets
    combined_sequences = []
    combined_labels = []
    alphabet = None
    
    for formula_id, formula, dataset, alph in all_data:
        if alphabet is None:
            alphabet = alph
        elif alphabet != alph:
            print(f"Warning: Alphabet mismatch. Using first alphabet: {alphabet}")
        
        for seq, label in dataset:
            combined_sequences.append(seq)
            combined_labels.append(label)
    
    print(f"\nCombined Dataset Statistics:")
    print(f"  Total sequences: {len(combined_sequences)}")
    print(f"  Alphabet: {alphabet}")
    print(f"  Positive examples: {sum(combined_labels)}")
    print(f"  Negative examples: {len(combined_labels) - sum(combined_labels)}")
    
    # Extract features
    print(f"\nExtracting features...")
    X = np.array([extract_features(seq, alphabet, sequence_length) for seq in combined_sequences])
    y = np.array(combined_labels)
    
    print(f"  Feature vector size: {X.shape[1]}")
    
    # Split into train and test
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"\nTrain/Test Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train logistic regression
    print(f"\nTraining logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC/ROC
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc_score = 0.0
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([1])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds),
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'alphabet': alphabet,
        'feature_size': X.shape[1]
    }


def main():
    """Main function to train on combined dataset from all formulas."""
    
    print("=" * 70)
    print("Logistic Regression Classification - Combined Dataset")
    print("=" * 70)
    
    # Load all datasets
    print("\nLoading datasets...")
    all_data = load_all_datasets()
    print(f"Loaded {len(all_data)} formulas with datasets")
    
    if len(all_data) == 0:
        print("Error: No datasets found. Please run generate_datasets.py first.")
        return
    
    # Train on combined dataset
    sequence_length = 10
    result = train_on_combined_dataset(all_data, sequence_length=sequence_length)
    
    # Print results
    print(f"\n{'='*70}")
    print("Evaluation Results on Combined Dataset")
    print(f"{'='*70}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1 Score:  {result['f1']:.4f}")
    print(f"  AUC-ROC:   {result['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  {result['confusion_matrix']}")
    
    print(f"\nDetailed Classification Report:")
    report = result['classification_report']
    print(f"  Class 0 (Negative):")
    print(f"    Precision: {report['0']['precision']:.4f}")
    print(f"    Recall:    {report['0']['recall']:.4f}")
    print(f"    F1-score:  {report['0']['f1-score']:.4f}")
    print(f"    Support:   {report['0']['support']}")
    print(f"  Class 1 (Positive):")
    print(f"    Precision: {report['1']['precision']:.4f}")
    print(f"    Recall:    {report['1']['recall']:.4f}")
    print(f"    F1-score:  {report['1']['f1-score']:.4f}")
    print(f"    Support:   {report['1']['support']}")
    
    # Plot ROC curve
    print(f"\nGenerating ROC curve plot...")
    plt.figure(figsize=(10, 8))
    
    fpr, tpr, _ = result['roc_curve']
    plt.plot(fpr, tpr, label=f"Combined Model (AUC={result['auc']:.3f})", 
            linewidth=3, color='blue')
    
    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC=0.5)', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Combined Dataset (All Formulas)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_plot_path = 'roc_curve_combined_dataset.png'
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved ROC curve plot to: {roc_plot_path}")
    plt.close()
    
    # Save results
    results_summary = {
        'dataset_info': {
            'num_formulas': len(all_data),
            'total_sequences': result['n_train'] + result['n_test'],
            'train_samples': result['n_train'],
            'test_samples': result['n_test'],
            'alphabet': result['alphabet'],
            'feature_size': result['feature_size'],
            'sequence_length': sequence_length
        },
        'metrics': {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'auc': float(result['auc'])
        },
        'confusion_matrix': result['confusion_matrix'].tolist(),
        'classification_report': result['classification_report']
    }
    
    results_path = 'logistic_regression_combined_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved detailed results to: {results_path}")
    
    return result


if __name__ == "__main__":
    main()

