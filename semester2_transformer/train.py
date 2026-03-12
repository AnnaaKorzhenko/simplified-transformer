import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformer import SimplifiedTransformer
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SequenceDataset(Dataset):
    
    def __init__(self, sequences, labels, symbol_to_idx, omega_idx):
        self.sequences = sequences
        self.labels = labels
        self.symbol_to_idx = symbol_to_idx
        self.omega_idx = omega_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert sequence to indices and add omega at the end
        seq = self.sequences[idx]
        indices = [self.symbol_to_idx[s] for s in seq]
        indices.append(self.omega_idx)  # Add omega symbol at the end
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


def load_data(csv_path, json_path):
    # Load dataset from csv file
    df = pd.read_csv(csv_path)
    
    # Load formula metadata
    with open(json_path, 'r') as f:
        formula_data = json.load(f)
    
    # Get sequences and labels
    sequences = []
    labels = []
    
    for _, row in df.iterrows():
        seq_str = row['sequence']
        seq = seq_str.split(',')
        sequences.append(seq)
        labels.append(int(row['label']))
    
    # Create symbol to index mapping
    alphabet = formula_data['metadata']['alphabet']
    # Add omega symbol at the end
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(alphabet)}
    omega_idx = len(alphabet)  
    vocab_size = len(alphabet) + 1  
    
    return sequences, labels, symbol_to_idx, omega_idx, vocab_size, formula_data


def create_splits(sequences, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    n_total = len(sequences)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_sequences = [sequences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_sequences = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)


def evaluate(model, dataloader, device, threshold=0.0):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            

            outputs = model(sequences)
            
            preds = outputs.float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        

        raw_output = model.forward_raw(sequences)
        

        loss = criterion(raw_output, labels)
        

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def plot_training_history(history, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', label='Accuracy', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs, history['val_precision'], 'b-', label='Precision', linewidth=2, marker='s', markersize=4)
    axes[0, 1].plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2, marker='^', markersize=4)
    axes[0, 1].plot(epochs, history['val_f1'], 'm-', label='F1 Score', linewidth=2, marker='d', markersize=4)
    axes[0, 1].set_title('Validation Metrics Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    ax2 = axes[1, 0]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    line2 = ax2_twin.plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2, marker='o', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12, color='b')
    ax2_twin.set_ylabel('Validation F1 Score', fontsize=12, color='r')
    ax2.set_title('Training Loss vs Validation F1', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2_twin.set_ylim([0, 1.05])
    
    axes[1, 1].plot(epochs, history['val_accuracy'], 'g-', label='Accuracy', linewidth=2)
    axes[1, 1].plot(epochs, history['val_precision'], 'b-', label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1'], 'm-', label='F1 Score', linewidth=2.5)
    axes[1, 1].set_title('All Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_dir}/training_history.png")
    plt.close()


def plot_confusion_matrix(cm, title='Confusion Matrix', save_path='plots/confusion_matrix.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def plot_final_metrics(test_metrics, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Final Test Set Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Final metrics plot saved to {save_dir}/final_metrics.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Simplified Transformer')
    parser.add_argument('--csv_path', type=str, default='/Users/victoriaportnaya/Downloads/dataset_33.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--json_path', type=str, default='/Users/victoriaportnaya/Downloads/formula_33.json',
                        help='Path to formula JSON file')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--d_prime', type=int, default=32, help='Query/Key/Value dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.0, help='Classification threshold')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    sequences, labels, symbol_to_idx, omega_idx, vocab_size, formula_data = load_data(
        args.csv_path, args.json_path
    )
    print(f"Loaded {len(sequences)} sequences")
    print(f"Vocabulary size: {vocab_size} (including omega)")
    print(f"Alphabet size: {vocab_size - 1}")
    
    print("Creating train/val/test splits...")
    (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels) = create_splits(
        sequences, labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    train_dataset = SequenceDataset(train_sequences, train_labels, symbol_to_idx, omega_idx)
    val_dataset = SequenceDataset(val_sequences, val_labels, symbol_to_idx, omega_idx)
    test_dataset = SequenceDataset(test_sequences, test_labels, symbol_to_idx, omega_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    

    print("Creating model...")
    model = SimplifiedTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_prime=args.d_prime,
        num_layers=args.num_layers,
        activation=torch.nn.functional.relu,
        threshold=args.threshold
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    print("\nStarting training...")
    best_val_f1 = 0.0
    

    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_metrics = evaluate(model, val_loader, device, args.threshold)
        
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print("  * Best model saved!")
        print()
    
    print("\nGenerating plots...")
    plot_training_history(history, save_dir=args.plot_dir)
    
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, test_loader, device, args.threshold)
    
    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative    {test_metrics['confusion_matrix'][0,0]:4d}      {test_metrics['confusion_matrix'][0,1]:4d}")
    print(f"        Positive    {test_metrics['confusion_matrix'][1,0]:4d}      {test_metrics['confusion_matrix'][1,1]:4d}")
    print("="*60)
    

    plot_final_metrics(test_metrics, save_dir=args.plot_dir)
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        title='Test Set Confusion Matrix',
        save_path=os.path.join(args.plot_dir, 'test_confusion_matrix.png')
    )
    
    print(f"\nAll plots saved to {args.plot_dir}/ directory")


if __name__ == '__main__':
    main()
