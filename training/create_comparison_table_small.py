"""
Create comparison table for small dataset results.
"""

import json
import os

def load_results(filepath):
    """Load results from JSON file"""
    # Go up one level from training/ to root directory
    if not os.path.isabs(filepath):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filepath)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    """Create comparison table"""
    
    # Load all results (from root directory)
    lr_results = load_results('logistic_regression_combined_results.json')
    hard_results = load_results('simplified_transformer_hard_results.json')
    soft_results = load_results('simplified_transformer_softmax_results.json')
    
    print("=" * 100)
    print("COMPARISON TABLE - SMALL DATASET")
    print("=" * 100)
    print("\nDataset: 15 formulas, alphabet_size=5, 20 positive + 20 negative per formula")
    print(f"Total sequences: 600 (480 train, 120 test)\n")
    
    print(f"{'Model':<50} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
    print("-" * 110)
    
    if lr_results:
        m = lr_results['metrics']
        print(f"{'Logistic Regression':<50} {m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['auc']:<12.4f}")
    
    if hard_results:
        m = hard_results['metrics']
        print(f"{'Simplified Transformer (Hard Attention)':<50} {m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['auc']:<12.4f}")
    
    if soft_results:
        m = soft_results['metrics']
        print(f"{'Simplified Transformer (Softmax)':<50} {m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['auc']:<12.4f}")
    
    # Save comparison to JSON
    comparison = {
        'dataset': 'small',
        'description': '15 formulas, alphabet_size=5, 20 positive + 20 negative per formula',
        'total_sequences': 600,
        'train_samples': 480,
        'test_samples': 120,
        'results': {}
    }
    
    if lr_results:
        comparison['results']['logistic_regression'] = lr_results['metrics']
    
    if hard_results:
        comparison['results']['transformer_hard'] = hard_results['metrics']
    
    if soft_results:
        comparison['results']['transformer_softmax'] = soft_results['metrics']
    
    # Save to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root_dir, 'comparison_table_small_dataset.json')
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n\nComparison saved to: {output_path}")
    
    # Print detailed breakdown
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    
    if lr_results:
        print("\nLogistic Regression:")
        print(f"  Confusion Matrix: {lr_results['confusion_matrix']}")
        print(f"  True Negatives: {lr_results['confusion_matrix'][0][0]}")
        print(f"  False Positives: {lr_results['confusion_matrix'][0][1]}")
        print(f"  False Negatives: {lr_results['confusion_matrix'][1][0]}")
        print(f"  True Positives: {lr_results['confusion_matrix'][1][1]}")
    
    if hard_results:
        print("\nSimplified Transformer (Hard Attention):")
        print(f"  Confusion Matrix: {hard_results['confusion_matrix']}")
        print(f"  True Negatives: {hard_results['confusion_matrix'][0][0]}")
        print(f"  False Positives: {hard_results['confusion_matrix'][0][1]}")
        print(f"  False Negatives: {hard_results['confusion_matrix'][1][0]}")
        print(f"  True Positives: {hard_results['confusion_matrix'][1][1]}")
    
    if soft_results:
        print("\nSimplified Transformer (Softmax):")
        print(f"  Confusion Matrix: {soft_results['confusion_matrix']}")
        print(f"  True Negatives: {soft_results['confusion_matrix'][0][0]}")
        print(f"  False Positives: {soft_results['confusion_matrix'][0][1]}")
        print(f"  False Negatives: {soft_results['confusion_matrix'][1][0]}")
        print(f"  True Positives: {soft_results['confusion_matrix'][1][1]}")


if __name__ == "__main__":
    main()


