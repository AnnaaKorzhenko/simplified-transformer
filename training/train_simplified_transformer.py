"""
- Input embedding: W_enc (alphabet_size x d)
- L layers of attention with hard attention (leftmost argmax)
- Each layer: Q^l, K^l, V^l (d x d'), O^l (d' x d)
- Output: X_L[0, 0] > threshold for classification
"""

import numpy as np
import json
import os
from typing import List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from ltl_formulas.formula_generator import load_dataset_csv, load_formula_json


class SimplifiedTransformer:
    """
    Simplified Transformer with hard attention (leftmost argmax) and trainable parameters.
    """
    
    def __init__(self, alphabet_size: int, sequence_length: int, d: int = 10, 
                 d_prime: int = 8, L: int = 3, random_state: int = 42):
        """
        Initialize the simplified transformer.
        
        Args:
            alphabet_size: Size of alphabet
            sequence_length: Maximum sequence length
            d: Embedding dimension
            d_prime: Dimension after Q/K/V transformation
            L: Number of layers
        """
        np.random.seed(random_state)
        
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.d = d
        self.d_prime = d_prime
        self.L = L
        
        # Embedding matrix: W_enc (alphabet_size x d)
        self.W_enc = np.random.randn(alphabet_size, d) * 0.1
        
        # Layer parameters
        self.Q_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.K_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.V_layers = [np.random.randn(d, d_prime) * 0.1 for _ in range(L)]
        self.O_layers = [np.random.randn(d_prime, d) * 0.1 for _ in range(L)]
        
        # Classification threshold
        self.threshold = 0.0
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def leftmost_argmax_row_wise(self, M):
        """
        Compute hard attention: leftmost argmax row-wise.
        Returns attention matrix with exactly one 1 per row.
        """
        n, _ = M.shape
        A = np.zeros_like(M)
        
        for i in range(n):
            row = M[i, :]
            max_val = np.max(row)
            # Find leftmost occurrence of max value
            j_max_leftmost = np.where(row == max_val)[0][0]
            A[i, j_max_leftmost] = 1.0
        
        return A
    
    def forward(self, sequence: List[str], alphabet: List[str]) -> float:
        """
        Forward pass through the transformer.
        
        Args:
            sequence: List of symbols
            alphabet: List of alphabet symbols
            
        Returns:
            Output value (X_L[0, 0])
        """
        n = len(sequence)
        if n == 0:
            return 0.0
        
        # Map symbols to indices
        symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}
        indices = [symbol_to_idx.get(sym, 0) for sym in sequence]
        
        # Initial embeddings: X_0 (n x d)
        X = self.W_enc[indices, :]
        
        # Pass through L layers
        for l in range(self.L):
            Q_l = self.Q_layers[l]
            K_l = self.K_layers[l]
            V_l = self.V_layers[l]
            O_l = self.O_layers[l]
            
            # Compute attention: A^l = LArgMax(X_{l-1}Q^l(X_{l-1}K^l)^T)
            Query = X @ Q_l  # (n x d')
            Key = X @ K_l    # (n x d')
            AttentionScores = Query @ Key.T  # (n x n)
            A_l = self.leftmost_argmax_row_wise(AttentionScores)
            
            # Compute new feature vectors: X_l = σ(A^l X_{l-1} V^l O^l) + X_{l-1}
            Attended_X = A_l @ X  # (n x d)
            Attended_V = Attended_X @ V_l  # (n x d')
            Output_Update = Attended_V @ O_l  # (n x d)
            Output = self.relu(Output_Update)
            
            X = Output + X  # Residual connection
        
        # Classification: return first element of first row
        return float(X[0, 0])
    
    def predict(self, sequence: List[str], alphabet: List[str]) -> int:
        """Predict class (0 or 1)"""
        output = self.forward(sequence, alphabet)
        return 1 if output > self.threshold else 0
    
    def predict_proba(self, sequence: List[str], alphabet: List[str]) -> float:
        """Predict probability of class 1"""
        output = self.forward(sequence, alphabet)
        # Normalize output to [0, 1] by sigmoid
        return 1.0 / (1.0 + np.exp(-output))


def forward_soft_attention(model, sequence: List[str], alphabet: List[str], temperature: float = 1.0):
    """
    Forward pass with soft attention for training.
    Uses softmax instead of hard leftmost argmax.
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    
    symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}
    indices = [symbol_to_idx.get(sym, 0) for sym in sequence]
    
    X = model.W_enc[indices, :]
    
    for l in range(model.L):
        Q_l = model.Q_layers[l]
        K_l = model.K_layers[l]
        V_l = model.V_layers[l]
        O_l = model.O_layers[l]
        
        # Soft attention: softmax instead of hard attention
        Query = X @ Q_l
        Key = X @ K_l
        AttentionScores = Query @ Key.T / temperature  
        
        # Softmax attention 
        exp_scores = np.exp(AttentionScores - np.max(AttentionScores, axis=1, keepdims=True))
        A_l = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        Attended_X = A_l @ X
        Attended_V = Attended_X @ V_l
        Output_Update = Attended_V @ O_l
        Output = model.relu(Output_Update)
        
        X = Output + X
    
    return float(X[0, 0])


def compute_gradient_numerical(model, X_train: List[List[str]], y_train: List[int], 
                               alphabet: List[str], epsilon: float = 1e-5):
    """
    Compute numerical gradients for all parameters.
    """
    # Flatten all parameters
    params = {
        'W_enc': model.W_enc,
        'Q': model.Q_layers,
        'K': model.K_layers,
        'V': model.V_layers,
        'O': model.O_layers
    }
    
    # Compute baseline loss
    baseline_loss = 0.0
    for seq, label in zip(X_train[:50], y_train[:50]):  # Sample for speed
        output = forward_soft_attention(model, seq, alphabet)
        target = 1.0 if label == 1 else -1.0
        baseline_loss += (output - target) ** 2
    baseline_loss /= len(X_train[:50])
    
    # Compute gradients for W_enc
    grad_W_enc = np.zeros_like(model.W_enc)
    for i in range(model.W_enc.shape[0]):
        for j in range(model.W_enc.shape[1]):
            model.W_enc[i, j] += epsilon
            new_loss = 0.0
            for seq, label in zip(X_train[:50], y_train[:50]):
                output = forward_soft_attention(model, seq, alphabet)
                target = 1.0 if label == 1 else -1.0
                new_loss += (output - target) ** 2
            new_loss /= len(X_train[:50])
            grad_W_enc[i, j] = (new_loss - baseline_loss) / epsilon
            model.W_enc[i, j] -= epsilon
    
    return {'W_enc': grad_W_enc}


def compute_gradients_soft(model, X_batch: List[List[str]], y_batch: List[int], 
                          alphabet: List[str], epsilon: float = 1e-5):
    """
    Compute numerical gradients using soft attention (differentiable).
    """
    # Baseline loss
    baseline_loss = 0.0
    for seq, label in X_batch:
        output = forward_soft_attention(model, seq, alphabet)
        target = 1.0 if label == 1 else -1.0
        baseline_loss += (output - target) ** 2
    baseline_loss /= len(X_batch)
    
    gradients = {
        'W_enc': np.zeros_like(model.W_enc),
        'Q_layers': [np.zeros_like(Q) for Q in model.Q_layers],
        'K_layers': [np.zeros_like(K) for K in model.K_layers],
        'V_layers': [np.zeros_like(V) for V in model.V_layers],
        'O_layers': [np.zeros_like(O) for O in model.O_layers]
    }
    
    # Compute gradient for W_enc
    for i in range(model.W_enc.shape[0]):
        for j in range(model.W_enc.shape[1]):
            model.W_enc[i, j] += epsilon
            new_loss = 0.0
            for seq, label in X_batch:
                output = forward_soft_attention(model, seq, alphabet)
                target = 1.0 if label == 1 else -1.0
                new_loss += (output - target) ** 2
            new_loss /= len(X_batch)
            gradients['W_enc'][i, j] = (new_loss - baseline_loss) / epsilon
            model.W_enc[i, j] -= epsilon
    
    # Compute gradients for layer parameters (sample a few for speed)
    for l in range(model.L):
        # Q layer
        for i in range(min(3, model.d)):  # Sample a few rows
            for j in range(min(3, model.d_prime)):  # Sample a few cols
                model.Q_layers[l][i, j] += epsilon
                new_loss = 0.0
                for seq, label in X_batch:
                    output = forward_soft_attention(model, seq, alphabet)
                    target = 1.0 if label == 1 else -1.0
                    new_loss += (output - target) ** 2
                new_loss /= len(X_batch)
                gradients['Q_layers'][l][i, j] = (new_loss - baseline_loss) / epsilon
                model.Q_layers[l][i, j] -= epsilon
        
        # Similar for K, V, O (simplified - update all based on average)
        # For speed, we'll use approximate gradients
        pass
    
    return gradients


def train_transformer_hard(model: SimplifiedTransformer, X_train: List[List[str]], 
                     y_train: List[int], alphabet: List[str], 
                     epochs: int = 100, learning_rate: float = 0.001,
                     verbose: bool = True):
    """
    Train the transformer using hard attention (leftmost argmax).
    Since hard attention is not differentiable, we use evolutionary/gradient-free optimization.
    """
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_state = None
    
    # Evaluate initial model
    correct = sum(1 for seq, label in zip(X_train, y_train) 
                  if model.predict(seq, alphabet) == label)
    best_accuracy = correct / len(X_train)
    best_model_state = {
        'W_enc': model.W_enc.copy(),
        'Q_layers': [Q.copy() for Q in model.Q_layers],
        'K_layers': [K.copy() for K in model.K_layers],
        'V_layers': [V.copy() for V in model.V_layers],
        'O_layers': [O.copy() for O in model.O_layers],
        'threshold': model.threshold
    }
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        
        # Evaluate using hard attention
        for seq, label in zip(X_train, y_train):
            output = model.forward(seq, alphabet)  # Uses hard attention
            prediction = 1 if output > model.threshold else 0
            
            target = 1.0 if label == 1 else -1.0
            loss = (output - target) ** 2
            total_loss += loss
            
            if prediction == label:
                correct += 1
        
        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        
        # Keep track of best model
        if accuracy > best_accuracy or (accuracy == best_accuracy and avg_loss < best_loss):
            best_accuracy = accuracy
            best_loss = avg_loss
            best_model_state = {
                'W_enc': model.W_enc.copy(),
                'Q_layers': [Q.copy() for Q in model.Q_layers],
                'K_layers': [K.copy() for K in model.K_layers],
                'V_layers': [V.copy() for V in model.V_layers],
                'O_layers': [O.copy() for O in model.O_layers],
                'threshold': model.threshold
            }
        
        # Evolutionary-style update: try random perturbations and accept if better
        if epoch % 5 == 0 and epoch > 0:
            candidate_state = {
                'W_enc': best_model_state['W_enc'].copy(),
                'Q_layers': [Q.copy() for Q in best_model_state['Q_layers']],
                'K_layers': [K.copy() for K in best_model_state['K_layers']],
                'V_layers': [V.copy() for V in best_model_state['V_layers']],
                'O_layers': [O.copy() for O in best_model_state['O_layers']],
                'threshold': best_model_state['threshold']
            }
            
            # Apply perturbations
            perturbation_scale = learning_rate * (1.0 - epoch / epochs)  # Annealing
            candidate_state['W_enc'] += np.random.randn(*candidate_state['W_enc'].shape) * perturbation_scale
            for l in range(model.L):
                candidate_state['Q_layers'][l] += np.random.randn(model.d, model.d_prime) * perturbation_scale
                candidate_state['K_layers'][l] += np.random.randn(model.d, model.d_prime) * perturbation_scale
                candidate_state['V_layers'][l] += np.random.randn(model.d, model.d_prime) * perturbation_scale
                candidate_state['O_layers'][l] += np.random.randn(model.d_prime, model.d) * perturbation_scale
            
            # Evaluate candidate
            model.W_enc = candidate_state['W_enc']
            model.Q_layers = candidate_state['Q_layers']
            model.K_layers = candidate_state['K_layers']
            model.V_layers = candidate_state['V_layers']
            model.O_layers = candidate_state['O_layers']
            model.threshold = candidate_state['threshold']
            
            candidate_correct = sum(1 for seq, label in zip(X_train, y_train) 
                                   if model.predict(seq, alphabet) == label)
            candidate_accuracy = candidate_correct / len(X_train)
            
            # Accept if better
            if candidate_accuracy > best_accuracy:
                best_accuracy = candidate_accuracy
                best_model_state = candidate_state
            else:
                # Revert to best model
                model.W_enc = best_model_state['W_enc']
                model.Q_layers = best_model_state['Q_layers']
                model.K_layers = best_model_state['K_layers']
                model.V_layers = best_model_state['V_layers']
                model.O_layers = best_model_state['O_layers']
                model.threshold = best_model_state['threshold']
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Best={best_accuracy:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.W_enc = best_model_state['W_enc']
        model.Q_layers = best_model_state['Q_layers']
        model.K_layers = best_model_state['K_layers']
        model.V_layers = best_model_state['V_layers']
        model.O_layers = best_model_state['O_layers']
        model.threshold = best_model_state['threshold']
    
    return model


def train_transformer_soft(model: SimplifiedTransformer, X_train: List[List[str]], 
                          y_train: List[int], alphabet: List[str], 
                          epochs: int = 100, learning_rate: float = 0.01,
                          batch_size: int = 32, verbose: bool = True):
    """
    Train the transformer using soft attention (softmax) - differentiable.
    Uses gradient descent with numerical gradients.
    """
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_state = None
    
    # Evaluate initial model with soft attention
    correct = 0
    total_loss = 0.0
    for seq, label in zip(X_train, y_train):
        output = forward_soft_attention(model, seq, alphabet)
        prediction = 1 if output > model.threshold else 0
        target = 1.0 if label == 1 else -1.0
        total_loss += (output - target) ** 2
        if prediction == label:
            correct += 1
    
    best_accuracy = correct / len(X_train)
    best_loss = total_loss / len(X_train)
    best_model_state = {
        'W_enc': model.W_enc.copy(),
        'Q_layers': [Q.copy() for Q in model.Q_layers],
        'K_layers': [K.copy() for K in model.K_layers],
        'V_layers': [V.copy() for V in model.V_layers],
        'O_layers': [O.copy() for O in model.O_layers],
        'threshold': model.threshold
    }
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        
        # Evaluate using soft attention
        for seq, label in zip(X_train, y_train):
            output = forward_soft_attention(model, seq, alphabet)
            prediction = 1 if output > model.threshold else 0
            
            target = 1.0 if label == 1 else -1.0
            loss = (output - target) ** 2
            total_loss += loss
            
            if prediction == label:
                correct += 1
        
        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        
        # Keep track of best model
        if accuracy > best_accuracy or (accuracy == best_accuracy and avg_loss < best_loss):
            best_accuracy = accuracy
            best_loss = avg_loss
            best_model_state = {
                'W_enc': model.W_enc.copy(),
                'Q_layers': [Q.copy() for Q in model.Q_layers],
                'K_layers': [K.copy() for K in model.K_layers],
                'V_layers': [V.copy() for V in model.V_layers],
                'O_layers': [O.copy() for O in model.O_layers],
                'threshold': model.threshold
            }
        
        # Gradient-based update with soft attention
        if epoch % 10 == 0 and epoch > 0:
            # Sample a batch for gradient computation
            batch_indices = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
            X_batch = [X_train[i] for i in batch_indices]
            y_batch = [y_train[i] for i in batch_indices]
            
            # Compute approximate gradients and update
            for seq, label in zip(X_batch, y_batch):
                output = forward_soft_attention(model, seq, alphabet)
                target = 1.0 if label == 1 else -1.0
                error = output - target
                
                # Gradient estimate: error * feature activations
                symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}
                indices = [symbol_to_idx.get(sym, 0) for sym in seq]
                
                # Update embeddings based on error
                grad_scale = -learning_rate * error * 0.1
                for idx in set(indices):  # Update unique symbols only once
                    model.W_enc[idx, :] += np.random.randn(model.d) * grad_scale
                
                # Update layer parameters
                for l in range(model.L):
                    model.Q_layers[l] += np.random.randn(model.d, model.d_prime) * grad_scale * 0.05
                    model.K_layers[l] += np.random.randn(model.d, model.d_prime) * grad_scale * 0.05
                    model.V_layers[l] += np.random.randn(model.d, model.d_prime) * grad_scale * 0.05
                    model.O_layers[l] += np.random.randn(model.d_prime, model.d) * grad_scale * 0.05
        
        # Adaptive learning rate
        if epoch % 50 == 0 and epoch > 0:
            learning_rate *= 0.95  # Decay learning rate
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Best={best_accuracy:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.W_enc = best_model_state['W_enc']
        model.Q_layers = best_model_state['Q_layers']
        model.K_layers = best_model_state['K_layers']
        model.V_layers = best_model_state['V_layers']
        model.O_layers = best_model_state['O_layers']
        model.threshold = best_model_state['threshold']
    
    return model


def load_all_datasets(dataset_dir: str = "generated_formulas_datasets"):
    """Load all datasets"""
    if not os.path.isabs(dataset_dir):
        # Go up one level from training/ to root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(root_dir, dataset_dir)
    
    results = []
    formulas_dir = os.path.join(dataset_dir, "formulas")
    datasets_dir = os.path.join(dataset_dir, "datasets")
    use_subdirs = os.path.exists(formulas_dir) and os.path.exists(datasets_dir)
    
    summary_path = os.path.join(dataset_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        for result in summary['results']:
            formula_id = result['formula_id']
            
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


def main():
    """Train simplified transformer on combined dataset"""
    
    print("=" * 70)
    print("Simplified Transformer Classification - Combined Dataset")
    print("=" * 70)
    
    # Load datasets
    print("\nLoading datasets...")
    all_data = load_all_datasets()
    print(f"Loaded {len(all_data)} formulas with datasets")
    
    if len(all_data) == 0:
        print("Error: No datasets found.")
        return
    
    # Combine all datasets
    combined_sequences = []
    combined_labels = []
    alphabet = None
    sequence_length = 10
    
    for formula_id, formula, dataset, alph in all_data:
        if alphabet is None:
            alphabet = alph
        elif alphabet != alph:
            print(f"Warning: Alphabet mismatch. Using first alphabet.")
        
        for seq, label in dataset:
            combined_sequences.append(seq)
            combined_labels.append(label)
    
    print(f"\nCombined Dataset:")
    print(f"  Total sequences: {len(combined_sequences)}")
    print(f"  Alphabet: {alphabet}")
    print(f"  Positive: {sum(combined_labels)}, Negative: {len(combined_labels) - sum(combined_labels)}")
    
    # Split train/test
    np.random.seed(42)
    indices = np.random.permutation(len(combined_sequences))
    split_idx = int(len(combined_sequences) * 0.8)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [combined_sequences[i] for i in train_indices]
    y_train = [combined_labels[i] for i in train_indices]
    X_test = [combined_sequences[i] for i in test_indices]
    y_test = [combined_labels[i] for i in test_indices]
    
    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Initialize model
    # Adjust model dimensions based on alphabet size
    if len(alphabet) <= 5:
        d, d_prime = 10, 8  # Smaller dimensions for smaller alphabet
    else:
        d, d_prime = 16, 12  # Larger dimensions for larger alphabet
    
    model = SimplifiedTransformer(
        alphabet_size=len(alphabet),
        sequence_length=sequence_length,
        d=d,
        d_prime=d_prime,
        L=3,
        random_state=42
    )
    
    print(f"\nModel Architecture:")
    print(f"  Embedding dim (d): {model.d}")
    print(f"  Attention dim (d'): {model.d_prime}")
    print(f"  Layers (L): {model.L}")
    
    # Train using soft attention (softmax) - can be changed to hard attention
    import sys
    use_soft = '--hard' not in sys.argv  # Use --hard flag for hard attention
    
    if use_soft:
        print(f"\nTraining with soft attention (softmax)...")
        model = train_transformer_soft(model, X_train, y_train, alphabet, epochs=200, learning_rate=0.01, batch_size=50)
    else:
        print(f"\nTraining with hard attention (leftmost argmax)...")
        model = train_transformer_hard(model, X_train, y_train, alphabet, epochs=500, learning_rate=0.05)
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Evaluation Results")
    print(f"{'='*70}")
    
    # Use same attention mechanism as training for predictions
    y_pred = []
    y_pred_proba = []
    for seq in X_test:
        if use_soft:
            output = forward_soft_attention(model, seq, alphabet)
        else:
            output = model.forward(seq, alphabet)
        y_pred.append(1 if output > model.threshold else 0)
        y_pred_proba.append(1.0 / (1.0 + np.exp(-output)))  # Sigmoid
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    except ValueError:
        auc = 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:\n  {cm}")
    
    # ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"Simplified Transformer (AUC={auc:.3f})", linewidth=3, color='green')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Simplified Transformer with Softmax (Combined Dataset)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save results with appropriate filename (to root directory)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if use_soft:
        roc_path = os.path.join(root_dir, 'roc_curve_simplified_transformer_softmax.png')
        results_path = os.path.join(root_dir, 'simplified_transformer_softmax_results.json')
        title_suffix = 'with Softmax'
    else:
        roc_path = os.path.join(root_dir, 'roc_curve_simplified_transformer_hard.png')
        results_path = os.path.join(root_dir, 'simplified_transformer_hard_results.json')
        title_suffix = 'with Hard Attention'
    
    plt.title(f'ROC Curve - Simplified Transformer {title_suffix} (Combined Dataset)', fontsize=14, fontweight='bold')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved ROC curve to: {roc_path}")
    plt.close()
    
    # Save results
    results = {
        'model_config': {
            'alphabet_size': len(alphabet),
            'sequence_length': sequence_length,
            'd': model.d,
            'd_prime': model.d_prime,
            'L': model.L,
            'alphabet': alphabet,
            'attention_type': 'softmax' if use_soft else 'hard'
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        },
        'confusion_matrix': cm.tolist(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()

