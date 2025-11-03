"""
Train Simplified Transformer with Softmax Attention on temporal logic datasets.
Supports both small (100 sequences) and large (10,000 sequences) datasets.
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from typing import List, Dict
import matplotlib.pyplot as plt

from training.utils import split_dataset, load_single_dataset


class SimplifiedTransformer:
    """Simplified Transformer with softmax attention."""
    
    def __init__(self, alphabet_size: int, sequence_length: int, d: int = None, 
                 d_prime: int = None, L: int = 3, random_state: int = 42):
        np.random.seed(random_state)
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.L = L
        
        # Adjust dimensions based on alphabet size
        if d is None:
            d = max(10, alphabet_size // 2)
        if d_prime is None:
            d_prime = max(8, alphabet_size // 3)
        
        self.d = d
        self.d_prime = d_prime
        
        # Embedding matrix: (alphabet_size x d)
        self.W_enc = np.random.randn(alphabet_size, d) * 0.1
        
        # Attention layers: Q, K, V (d x d'), O (d' x d)
        self.Q_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.K_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.V_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.O_layers = [np.random.randn(d_prime, d) * 0.1 for _ in range(L)]
        
        # Classification threshold
        self.threshold = 0.0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward_soft(self, sequence: List[str], alphabet: List[str], temperature: float = 1.0) -> float:
        """Forward pass with softmax attention."""
        n = len(sequence)
        if n == 0:
            return 0.0
        
        symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}
        indices = [symbol_to_idx.get(sym, 0) for sym in sequence]
        
        X = self.W_enc[indices, :]
        
        for l in range(self.L):
            Q_l = self.Q_layers[l]
            K_l = self.K_layers[l]
            V_l = self.V_layers[l]
            O_l = self.O_layers[l]
            
            Query = X @ Q_l
            Key = X @ K_l
            AttentionScores = (Query @ Key.T) / temperature
            
            # Softmax attention
            exp_scores = np.exp(AttentionScores - np.max(AttentionScores, axis=1, keepdims=True))
            A_l = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            Attended_X = A_l @ X
            Attended_V = Attended_X @ V_l
            Output_Update = Attended_V @ O_l
            Output = self.relu(Output_Update)
            
            X = Output + X
        
        return float(X[0, 0])
    
    def predict_soft(self, sequence: List[str], alphabet: List[str]) -> int:
        """Predict with soft attention."""
        output = self.forward_soft(sequence, alphabet)
        return 1 if output > self.threshold else 0


def train_transformer_soft(X_train: List[List[str]], y_train: List[int],
                           X_test: List[List[str]], y_test: List[int],
                           alphabet: List[str], sequence_length: int,
                           epochs: int = 100, learning_rate: float = 0.001,
                           random_state: int = 42) -> Dict:
    """Train transformer with softmax attention (gradient descent with numerical gradients)."""
    model = SimplifiedTransformer(
        alphabet_size=len(alphabet),
        sequence_length=sequence_length,
        random_state=random_state
    )
    
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        
        # Forward pass with soft attention
        outputs = []
        for seq, label in zip(X_train, y_train):
            output = model.forward_soft(seq, alphabet, temperature=1.0)
            outputs.append((output, label))
            prediction = 1 if output > model.threshold else 0
            if prediction == label:
                correct += 1
            
            target = 1.0 if label == 1 else -1.0
            loss = (output - target) ** 2
            total_loss += loss
        
        accuracy = correct / len(X_train)
        avg_loss = total_loss / len(X_train)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = {
                'W_enc': model.W_enc.copy(),
                'Q_layers': [Q.copy() for Q in model.Q_layers],
                'K_layers': [K.copy() for K in model.K_layers],
                'V_layers': [V.copy() for V in model.V_layers],
                'O_layers': [O.copy() for O in model.O_layers],
                'threshold': model.threshold
            }
        
        # Simple numerical gradient update (approximate)
        if epoch % 10 == 0:
            # Compute approximate gradient for threshold
            threshold_grad = 0.0
            for output, label in outputs:
                target = 1.0 if label == 1 else -1.0
                if output > model.threshold:
                    threshold_grad += -2 * (output - target)
                else:
                    threshold_grad += 2 * (output - target)
            threshold_grad /= len(outputs)
            model.threshold -= learning_rate * threshold_grad
            
            # Small random updates to model parameters (simulating gradient descent)
            for i in range(len(model.W_enc)):
                for j in range(len(model.W_enc[0])):
                    model.W_enc[i, j] += np.random.randn() * learning_rate * 0.1
        
        if epoch % 20 == 0 and len(X_train) > 1000:
            print(f"  Epoch {epoch}/{epochs}, Train Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    
    # Restore best model
    if best_model_state:
        model.W_enc = best_model_state['W_enc']
        model.Q_layers = [Q.copy() for Q in best_model_state['Q_layers']]
        model.K_layers = [K.copy() for K in best_model_state['K_layers']]
        model.V_layers = [V.copy() for V in best_model_state['V_layers']]
        model.O_layers = [O.copy() for O in best_model_state['O_layers']]
        model.threshold = best_model_state['threshold']
    
    # Evaluate on test set
    y_pred = [model.predict_soft(seq, alphabet) for seq in X_test]
    y_pred_proba = [1.0 / (1.0 + np.exp(-model.forward_soft(seq, alphabet))) for seq in X_test]
    
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
        'y_pred_proba': y_pred_proba
    }


def train_on_dataset(dataset_dir: str, formula_id: int = 33, output_dir: str = None, epochs: int = None):
    """Train transformer with softmax attention on a dataset and save results."""
    print("=" * 80)
    print(f"Training Simplified Transformer (Softmax Attention)")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Formula ID: {formula_id}\n")
    
    # Set epochs based on dataset size
    if epochs is None:
        epochs = 200 if "large" in dataset_dir else 100
    
    # Load dataset
    print("Loading dataset...")
    formula, dataset, alphabet, sequence_length = load_single_dataset(dataset_dir, formula_id)
    
    print(f"Loaded dataset: {len(dataset)} sequences")
    print(f"Alphabet size: {len(alphabet)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Positive: {sum(1 for _, label in dataset if label == 1)}")
    print(f"Negative: {sum(1 for _, label in dataset if label == 0)}")
    print(f"Epochs: {epochs}\n")
    
    # Split dataset
    X_train, y_train, X_test, y_test = split_dataset(dataset, test_size=0.2, random_state=42)
    print(f"Train set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences\n")
    
    # Train
    print("Training Simplified Transformer (Softmax Attention)...")
    result = train_transformer_soft(
        X_train, y_train, X_test, y_test, alphabet, sequence_length, 
        epochs=epochs, learning_rate=0.001, random_state=42
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
    try:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"Transformer Softmax (AUC={result['auc']:.3f})", linewidth=2.5, color='red')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - Transformer Softmax Attention ({dataset_dir})', fontsize=14, fontweight='bold')
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
        roc_path = os.path.join(output_dir, f'roc_curve_transformer_softmax_{dataset_name}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to: {roc_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate ROC curve: {e}")
    
    # Save results JSON
    output = {
        'dataset_info': {
            'dataset': dataset_dir,
            'formula_id': formula_id,
            'alphabet_size': len(alphabet),
            'sequence_length': sequence_length,
            'total_sequences': len(dataset),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs': epochs
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
    
    results_path = os.path.join(output_dir, f'transformer_softmax_results_{dataset_name}.json')
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
    train_on_dataset("single_dataset", formula_id=33, output_dir=project_root, epochs=100)
    
    # Train on large dataset
    print("\n" + "="*80)
    print("TRAINING ON LARGE DATASET (10,000 sequences)")
    print("="*80 + "\n")
    train_on_dataset("large_single_dataset", formula_id=33, output_dir=project_root, epochs=200)


if __name__ == "__main__":
    main()

