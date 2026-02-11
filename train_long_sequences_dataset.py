"""
Train all models (Logistic Regression, Transformer Hard, Transformer Softmax) 
on the long sequences dataset (formula 35).
"""

import os
import json
import sys

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from train_logistic_regression import train_on_dataset as train_lr
from train_transformer_hard import train_on_dataset as train_transformer_hard
from train_transformer_softmax import train_on_dataset as train_transformer_softmax

def main():
    """Train all three models on the long sequences dataset."""
    
    dataset_dir = "generator_updated/dataset"
    formula_id = 35  # Long sequences formula
    
    print("=" * 80)
    print("TRAINING ALL MODELS ON LONG SEQUENCES DATASET")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Formula ID: {formula_id} (Complex: 4 disjunctions, 3 conjunctions, length 20)")
    print("=" * 80)
    print()
    
    # Check if dataset exists
    formula_file = os.path.join(dataset_dir, f"formula_{formula_id}.json")
    dataset_file = os.path.join(dataset_dir, f"dataset_{formula_id}.csv")
    
    if not os.path.exists(formula_file):
        print(f"Error: Formula file not found: {formula_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        return
    
    results = {}
    
    # 1. Train Logistic Regression
    print("\n" + "=" * 80)
    print("1. TRAINING LOGISTIC REGRESSION")
    print("=" * 80)
    try:
        lr_result = train_lr(dataset_dir, formula_id, output_dir="generator_updated/results")
        results['logistic_regression'] = {
            'accuracy': lr_result['accuracy'],
            'precision': lr_result['precision'],
            'recall': lr_result['recall'],
            'f1': lr_result['f1'],
            'auc': lr_result['auc'],
            'confusion_matrix': lr_result['confusion_matrix']
        }
        print("✓ Logistic Regression training completed")
    except Exception as e:
        print(f"✗ Error training Logistic Regression: {e}")
        import traceback
        traceback.print_exc()
        results['logistic_regression'] = {'error': str(e)}
    
    # 2. Train Transformer with Hard Attention
    print("\n" + "=" * 80)
    print("2. TRAINING TRANSFORMER WITH HARD ATTENTION")
    print("=" * 80)
    try:
        hard_result = train_transformer_hard(
            dataset_dir, formula_id, output_dir="generator_updated/results", epochs=200
        )
        results['transformer_hard'] = {
            'accuracy': hard_result['accuracy'],
            'precision': hard_result['precision'],
            'recall': hard_result['recall'],
            'f1': hard_result['f1'],
            'auc': hard_result['auc'],
            'confusion_matrix': hard_result['confusion_matrix']
        }
        print("✓ Transformer Hard Attention training completed")
    except Exception as e:
        print(f"✗ Error training Transformer Hard: {e}")
        import traceback
        traceback.print_exc()
        results['transformer_hard'] = {'error': str(e)}
    
    # 3. Train Transformer with Softmax Attention
    print("\n" + "=" * 80)
    print("3. TRAINING TRANSFORMER WITH SOFTMAX ATTENTION")
    print("=" * 80)
    try:
        soft_result = train_transformer_softmax(
            dataset_dir, formula_id, output_dir="generator_updated/results", epochs=200
        )
        results['transformer_softmax'] = {
            'accuracy': soft_result['accuracy'],
            'precision': soft_result['precision'],
            'recall': soft_result['recall'],
            'f1': soft_result['f1'],
            'auc': soft_result['auc'],
            'confusion_matrix': soft_result['confusion_matrix']
        }
        print("✓ Transformer Softmax Attention training completed")
    except Exception as e:
        print(f"✗ Error training Transformer Softmax: {e}")
        import traceback
        traceback.print_exc()
        results['transformer_softmax'] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RESULTS - LONG SEQUENCES DATASET")
    print("=" * 80)
    
    models = [
        ('Logistic Regression', 'logistic_regression'),
        ('Transformer (Hard)', 'transformer_hard'),
        ('Transformer (Softmax)', 'transformer_softmax')
    ]
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 85)
    
    for model_name, key in models:
        if key in results and 'error' not in results[key]:
            r = results[key]
            print(f"{model_name:<25} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                  f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")
        else:
            error_msg = results.get(key, {}).get('error', 'Not trained')
            print(f"{model_name:<25} ERROR: {error_msg}")
    
    # Save combined results
    os.makedirs("generator_updated/results", exist_ok=True)
    results_file = "generator_updated/results/all_models_results_long_sequences.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

