"""
Train all models on the smaller dataset and create comparison table.
"""

import json
import subprocess
import sys

def run_training(script_name, output_file):
    """Run a training script and capture results"""
    import os
    
    # Scripts are in the same directory (training/)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    print(f"\n{'='*70}")
    print(f"Running {script_name}...")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Run from root directory
    )
    
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        return None
    
    # Try to load results (from root directory)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root_dir, output_file) if not os.path.isabs(output_file) else output_file
    try:
        with open(output_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Results file {output_path} not found")
        return None


def main():
    """Train all models and create comparison"""
    
    print("=" * 70)
    print("Training All Models on Small Dataset")
    print("=" * 70)
    print("\nSmall Dataset: 15 formulas, alphabet_size=5, 20 positive sequences each")
    
    results = {}
    
    # 1. Logistic Regression
    print("\n\n" + "="*70)
    print("1. LOGISTIC REGRESSION")
    print("="*70)
    logreg_results = run_training(
        'train_logistic_regression.py',
        'logistic_regression_combined_results.json'
    )
    if logreg_results:
        results['logistic_regression'] = logreg_results['metrics']
    
    # 2. Simplified Transformer (Hard Attention)
    print("\n\n" + "="*70)
    print("2. SIMPLIFIED TRANSFORMER (HARD ATTENTION)")
    print("="*70)
    # Update script to use hard attention
    with open('train_simplified_transformer.py', 'r') as f:
        content = f.read()
    
    # Temporarily modify to use hard attention
    content_modified = content.replace(
        'train_transformer_soft(model, X_train, y_train, alphabet, epochs=200',
        'train_transformer(model, X_train, y_train, alphabet, epochs=500'
    ).replace(
        'Training with soft attention (softmax)',
        'Training with hard attention (leftmost argmax)'
    ).replace(
        'forward_soft_attention(model, seq, alphabet)',
        'model.forward(seq, alphabet)'
    ).replace(
        'roc_curve_simplified_transformer_softmax.png',
        'roc_curve_simplified_transformer_hard.png'
    ).replace(
        'simplified_transformer_softmax_results.json',
        'simplified_transformer_hard_results.json'
    )
    
    with open('train_simplified_transformer_temp.py', 'w') as f:
        f.write(content_modified)
    
    hard_results = run_training(
        'train_simplified_transformer_temp.py',
        'simplified_transformer_hard_results.json'
    )
    if hard_results:
        results['transformer_hard'] = hard_results['metrics']
    
    # 3. Simplified Transformer (Softmax)
    print("\n\n" + "="*70)
    print("3. SIMPLIFIED TRANSFORMER (SOFTMAX)")
    print("="*70)
    # Restore softmax version
    content_softmax = content.replace(
        'train_transformer(model, X_train, y_train, alphabet, epochs=500',
        'train_transformer_soft(model, X_train, y_train, alphabet, epochs=200'
    ).replace(
        'Training with hard attention (leftmost argmax)',
        'Training with soft attention (softmax)'
    )
    
    with open('train_simplified_transformer_temp.py', 'w') as f:
        f.write(content_softmax)
    
    soft_results = run_training(
        'train_simplified_transformer_temp.py',
        'simplified_transformer_softmax_results.json'
    )
    if soft_results:
        results['transformer_softmax'] = soft_results['metrics']
    
    # Cleanup
    import os
    if os.path.exists('train_simplified_transformer_temp.py'):
        os.remove('train_simplified_transformer_temp.py')
    
    # Print comparison table
    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE - SMALL DATASET")
    print("=" * 70)
    print("\nDataset: 15 formulas, alphabet_size=5, 20 positive + 20 negative per formula")
    print(f"Total sequences: ~600 (400 train, 200 test)\n")
    
    print(f"{'Model':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
    print("-" * 100)
    
    if 'logistic_regression' in results:
        lr = results['logistic_regression']
        print(f"{'Logistic Regression':<40} {lr['accuracy']:<12.4f} {lr['precision']:<12.4f} {lr['recall']:<12.4f} {lr['f1']:<12.4f} {lr['auc']:<12.4f}")
    
    if 'transformer_hard' in results:
        th = results['transformer_hard']
        print(f"{'Simplified Transformer (Hard)':<40} {th['accuracy']:<12.4f} {th['precision']:<12.4f} {th['recall']:<12.4f} {th['f1']:<12.4f} {th['auc']:<12.4f}")
    
    if 'transformer_softmax' in results:
        ts = results['transformer_softmax']
        print(f"{'Simplified Transformer (Softmax)':<40} {ts['accuracy']:<12.4f} {ts['precision']:<12.4f} {ts['recall']:<12.4f} {ts['f1']:<12.4f} {ts['auc']:<12.4f}")
    
    # Save comparison to root directory
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comparison = {
        'dataset': 'small',
        'description': '15 formulas, alphabet_size=5, 20 positive + 20 negative per formula',
        'results': results
    }
    
    output_path = os.path.join(root_dir, 'comparison_small_dataset.json')
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()


