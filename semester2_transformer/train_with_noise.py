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
import copy


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, symbol_to_idx, omega_idx, noise_level=0.0, 
                 noise_type='replacement', alphabet=None):
        self.sequences = sequences
        self.labels = labels
        self.symbol_to_idx = symbol_to_idx
        self.omega_idx = omega_idx
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.alphabet = alphabet
        
    def __len__(self):
        return len(self.sequences)
    
    def add_noise_replacement(self, seq):
        """Noise type 1: Replace symbols with random ones"""
        if self.noise_level == 0.0 or self.alphabet is None:
            return seq
        
        noisy_seq = list(seq)
        seq_length = len(noisy_seq)
        num_noise = max(1, int(seq_length * self.noise_level))
        
        noise_positions = np.random.choice(seq_length, size=min(num_noise, seq_length), replace=False)
        available_symbols = [s for s in self.alphabet if s != 'ω']
        
        for pos in noise_positions:
            noisy_seq[pos] = np.random.choice(available_symbols)
        
        return noisy_seq
    
    def add_noise_deletion(self, seq):
        """Noise type 2: Delete some symbols (replace with random to maintain length)"""
        if self.noise_level == 0.0 or self.alphabet is None:
            return seq
        
        noisy_seq = list(seq)
        seq_length = len(noisy_seq)
        num_noise = max(1, int(seq_length * self.noise_level))
        
        noise_positions = np.random.choice(seq_length, size=min(num_noise, seq_length), replace=False)
        available_symbols = [s for s in self.alphabet if s != 'ω']
        
        for pos in noise_positions:
            # "Delete" by replacing with random symbol (maintains sequence length)
            noisy_seq[pos] = np.random.choice(available_symbols)
        
        return noisy_seq
    
    def add_noise_insertion(self, seq):
        """Noise type 3: Insert random symbols (but we maintain length, so replace)"""
        return self.add_noise_replacement(seq)
    
    def add_noise_swap(self, seq):
        """Noise type 4: Swap adjacent symbols"""
        if self.noise_level == 0.0:
            return seq
        
        noisy_seq = list(seq)
        seq_length = len(noisy_seq)
        num_swaps = max(1, int(seq_length * self.noise_level))
        
        for _ in range(num_swaps):
            if seq_length < 2:
                break
            # Pick a random position (can't swap last position)
            pos = np.random.randint(0, seq_length - 1)
            # Swap with next position
            noisy_seq[pos], noisy_seq[pos + 1] = noisy_seq[pos + 1], noisy_seq[pos]
        
        return noisy_seq
    
    def add_noise_shuffle(self, seq):
        """Noise type 5: Randomly shuffle some positions"""
        if self.noise_level == 0.0:
            return seq
        
        noisy_seq = list(seq)
        seq_length = len(noisy_seq)
        num_shuffle = max(2, int(seq_length * self.noise_level))
        
        shuffle_positions = np.random.choice(seq_length, size=min(num_shuffle, seq_length), replace=False)
        values_to_shuffle = [noisy_seq[i] for i in shuffle_positions]
        np.random.shuffle(values_to_shuffle)
        for i, pos in enumerate(shuffle_positions):
            noisy_seq[pos] = values_to_shuffle[i]
        
        return noisy_seq
    
    def add_noise_mixed(self, seq):
        """Noise type 6: Mix of replacement and swap"""
        if self.noise_level == 0.0:
            return seq
        
        noisy_seq = self.add_noise_replacement(seq)
        original_level = self.noise_level
        self.noise_level = self.noise_level * 0.5
        noisy_seq = self.add_noise_swap(noisy_seq)
        self.noise_level = original_level
        
        return noisy_seq
    
    def add_noise(self, seq):
        """Add noise based on noise_type"""
        if self.noise_level == 0.0:
            return seq
        
        noise_functions = {
            'replacement': self.add_noise_replacement,
            'deletion': self.add_noise_deletion,
            'insertion': self.add_noise_insertion,
            'swap': self.add_noise_swap,
            'shuffle': self.add_noise_shuffle,
            'mixed': self.add_noise_mixed
        }
        
        noise_func = noise_functions.get(self.noise_type, self.add_noise_replacement)
        return noise_func(seq)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        if self.noise_level > 0.0:
            seq = self.add_noise(seq)
        
        indices = [self.symbol_to_idx[s] for s in seq]
        indices.append(self.omega_idx)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


def load_data(csv_path, json_path):
    df = pd.read_csv(csv_path)
    
    with open(json_path, 'r') as f:
        formula_data = json.load(f)
    
    sequences = []
    labels = []
    
    for _, row in df.iterrows():
        seq_str = row['sequence']
        seq = seq_str.split(',')
        sequences.append(seq)
        labels.append(int(row['label']))
    
    alphabet = formula_data['metadata']['alphabet']
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(alphabet)}
    omega_idx = len(alphabet)
    vocab_size = len(alphabet) + 1
    
    return sequences, labels, symbol_to_idx, omega_idx, vocab_size, formula_data, alphabet


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


def train_epoch(model, dataloader, criterion, optimizer, device, verbose=False):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        raw_output = model.forward_raw(sequences)
        loss = criterion(raw_output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        if verbose and (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}", end='\r')
    
    if verbose:
        print()  # New line after progress
    return total_loss / num_batches


def train_model(sequences, labels, symbol_to_idx, omega_idx, alphabet, vocab_size,
                noise_level, noise_type, epochs, batch_size, lr, device, threshold=0.0,
                early_stop_patience=5, min_delta=0.001):
    
    (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels) = create_splits(
        sequences, labels, 0.7, 0.15, 0.15, 42
    )
    
    train_dataset = SequenceDataset(train_sequences, train_labels, symbol_to_idx, omega_idx, 
                                   noise_level, noise_type, alphabet)
    val_dataset = SequenceDataset(val_sequences, val_labels, symbol_to_idx, omega_idx, 0.0, 'replacement', alphabet)
    test_dataset = SequenceDataset(test_sequences, test_labels, symbol_to_idx, omega_idx, 0.0, 'replacement', alphabet)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    model = SimplifiedTransformer(
        vocab_size=vocab_size,
        d_model=64,
        d_prime=32,
        num_layers=3,
        threshold=threshold
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    print(f"  Training for up to {epochs} epochs (early stop patience: {early_stop_patience})...")
    for epoch in range(epochs):
        print(f"  Epoch {epoch + 1}/{epochs}:", end=' ', flush=True)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, verbose=False)
        print(f"Train Loss: {train_loss:.4f}", end=' | ', flush=True)
        
        val_metrics = evaluate(model, val_loader, device, threshold)
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}", end='')
        
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1 + min_delta:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(" ✓ (best)", flush=True)
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{early_stop_patience})", flush=True)
        
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, threshold)
    
    return test_metrics, history, model


def plot_comparison(results, save_dir='plots_noise', compare_noise_types=False):
    os.makedirs(save_dir, exist_ok=True)
    
    noise_types = sorted(set(r['noise_type'] for r in results.values()))
    noise_levels = sorted(set(r['noise_level'] for r in results.values()))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    type_colors = {
        'replacement': '#3498db',
        'deletion': '#e74c3c',
        'insertion': '#2ecc71',
        'swap': '#f39c12',
        'shuffle': '#9b59b6',
        'mixed': '#e67e22'
    }
    
    def get_result(noise_type, noise_level):
        key = f"{noise_type}_{noise_level}"
        return results.get(key, None)
    
    if len(noise_types) == 1:
        noise_type = noise_types[0]
        
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            values = [get_result(noise_type, nl)[metric] for nl in noise_levels 
                     if get_result(noise_type, nl) is not None]
            valid_levels = [nl for nl in noise_levels if get_result(noise_type, nl) is not None]
            
            axes[idx].plot(valid_levels, values, 'o-', linewidth=2.5, markersize=10, 
                          color=colors[idx], markerfacecolor=colors[idx], markeredgecolor='white', markeredgewidth=2)
            axes[idx].set_title(f'{metric.capitalize()} vs Noise Level ({noise_type})', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Noise Level', fontsize=12)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].set_ylim([0, 1.05])
            if valid_levels:
                axes[idx].set_xlim([min(valid_levels) - 0.05, max(valid_levels) + 0.05])
            

            for i, (nl, val) in enumerate(zip(valid_levels, values)):
                axes[idx].annotate(f'{val:.3f}', (nl, val), textcoords="offset points", 
                                 xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noise_comparison_individual.png'), dpi=300, bbox_inches='tight')
        print(f"Individual metric plots saved to {save_dir}/noise_comparison_individual.png")
        plt.close()
    
    if len(noise_types) > 1:
        # Plot comparing noise types
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            for noise_type in noise_types:
                values = [get_result(noise_type, nl)[metric] for nl in noise_levels 
                         if get_result(noise_type, nl) is not None]
                valid_levels = [nl for nl in noise_levels if get_result(noise_type, nl) is not None]
                
                if values:
                    color = type_colors.get(noise_type, '#000000')
                    axes[idx].plot(valid_levels, values, 'o-', linewidth=2.5, markersize=8, 
                                  label=noise_type.capitalize(), color=color, markerfacecolor=color,
                                  markeredgecolor='white', markeredgewidth=2)
            
            axes[idx].set_title(f'{metric.capitalize()} vs Noise Level', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Noise Level', fontsize=12)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noise_types_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Noise types comparison plot saved to {save_dir}/noise_types_comparison.png")
        plt.close()
    
    # 2. All metrics on one plot (for first noise type or all if multiple)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    if len(noise_types) == 1:
        noise_type = noise_types[0]
        for idx, metric in enumerate(metrics):
            values = [get_result(noise_type, nl)[metric] for nl in noise_levels 
                     if get_result(noise_type, nl) is not None]
            valid_levels = [nl for nl in noise_levels if get_result(noise_type, nl) is not None]
            if values:
                ax.plot(valid_levels, values, 'o-', linewidth=2.5, markersize=10, 
                       label=metric.capitalize(), color=colors_metrics[idx], markerfacecolor=colors_metrics[idx], 
                       markeredgecolor='white', markeredgewidth=2)
    else:
        # Show F1 for all noise types
        for noise_type in noise_types:
            values = [get_result(noise_type, nl)['f1'] for nl in noise_levels 
                     if get_result(noise_type, nl) is not None]
            valid_levels = [nl for nl in noise_levels if get_result(noise_type, nl) is not None]
            if values:
                color = type_colors.get(noise_type, '#000000')
                ax.plot(valid_levels, values, 'o-', linewidth=2.5, markersize=10, 
                       label=f'{noise_type.capitalize()} (F1)', color=color, markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Noise Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('All Metrics vs Noise Level' if len(noise_types) == 1 else 'F1 Score Comparison Across Noise Types', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    if noise_levels:
        ax.set_xlim([min(noise_levels) - 0.05, max(noise_levels) + 0.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_comparison_all_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"All metrics plot saved to {save_dir}/noise_comparison_all_metrics.png")
    plt.close()
    
    # 3. Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(metrics))
    
    if len(noise_types) == 1:
        noise_type = noise_types[0]
        width = 0.8 / len(noise_levels)
        for i, noise_level in enumerate(noise_levels):
            result = get_result(noise_type, noise_level)
            if result:
                values = [result[metric] for metric in metrics]
                offset = (i - len(noise_levels)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=f'Noise {noise_level:.2f}', 
                             alpha=0.85, edgecolor='black', linewidth=1.5)
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        # Compare noise types at a specific level (use middle level)
        mid_level = noise_levels[len(noise_levels)//2] if noise_levels else 0.2
        width = 0.8 / len(noise_types)
        for i, noise_type in enumerate(noise_types):
            result = get_result(noise_type, mid_level)
            if result:
                values = [result[metric] for metric in metrics]
                offset = (i - len(noise_types)/2 + 0.5) * width
                color = type_colors.get(noise_type, '#000000')
                bars = ax.bar(x + offset, values, width, label=noise_type.capitalize(), 
                             alpha=0.85, edgecolor='black', linewidth=1.5, color=color)
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    title = 'Metrics Comparison Across Noise Levels' if len(noise_types) == 1 else f'Metrics Comparison Across Noise Types (level={mid_level:.2f})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_bar_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Bar comparison plot saved to {save_dir}/noise_bar_comparison.png")
    plt.close()
    
    # 4. Performance degradation plot
    baseline_key = None
    for key in results.keys():
        if results[key]['noise_level'] == 0.0:
            baseline_key = key
            break
    
    if baseline_key:
        baseline = results[baseline_key]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute degradation (for baseline noise type)
        baseline_type = baseline['noise_type']
        noise_levels_no_baseline = [nl for nl in noise_levels if nl > 0]
        acc_degradation = [get_result(baseline_type, nl)['accuracy'] - baseline['accuracy'] 
                          for nl in noise_levels_no_baseline if get_result(baseline_type, nl)]
        f1_degradation = [get_result(baseline_type, nl)['f1'] - baseline['f1'] 
                         for nl in noise_levels_no_baseline if get_result(baseline_type, nl)]
        valid_levels = [nl for nl in noise_levels_no_baseline if get_result(baseline_type, nl)]
        
        axes[0].plot(valid_levels, acc_degradation, 'o-', linewidth=2.5, 
                    markersize=10, label='Accuracy', color='#e74c3c', markerfacecolor='#e74c3c',
                    markeredgecolor='white', markeredgewidth=2)
        axes[0].plot(valid_levels, f1_degradation, 's-', linewidth=2.5, 
                    markersize=10, label='F1 Score', color='#9b59b6', markerfacecolor='#9b59b6',
                    markeredgecolor='white', markeredgewidth=2)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Performance Change', fontsize=12, fontweight='bold')
        axes[0].set_title('Performance Degradation from Baseline', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Percentage degradation
        acc_pct = [(get_result(baseline_type, nl)['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100 
                  for nl in valid_levels]
        f1_pct = [(get_result(baseline_type, nl)['f1'] - baseline['f1']) / baseline['f1'] * 100 
                 for nl in valid_levels]
        
        axes[1].plot(valid_levels, acc_pct, 'o-', linewidth=2.5, 
                    markersize=10, label='Accuracy', color='#e74c3c', markerfacecolor='#e74c3c',
                    markeredgecolor='white', markeredgewidth=2)
        axes[1].plot(valid_levels, f1_pct, 's-', linewidth=2.5, 
                    markersize=10, label='F1 Score', color='#9b59b6', markerfacecolor='#9b59b6',
                    markeredgecolor='white', markeredgewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Performance Change (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Percentage Performance Degradation', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (nl, acc, f1) in enumerate(zip(valid_levels, acc_pct, f1_pct)):
            axes[1].annotate(f'{acc:.1f}%', (nl, acc), textcoords="offset points", 
                           xytext=(0,12), ha='center', fontsize=9, fontweight='bold')
            axes[1].annotate(f'{f1:.1f}%', (nl, f1), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noise_degradation.png'), dpi=300, bbox_inches='tight')
        print(f"Degradation plot saved to {save_dir}/noise_degradation.png")
        plt.close()
    
    # 5. Heatmap of metrics
    if len(noise_types) == 1:
        noise_type = noise_types[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        metric_matrix = np.array([[get_result(noise_type, nl)[metric] for nl in noise_levels 
                                  if get_result(noise_type, nl)] for metric in metrics])
        valid_levels_hm = [nl for nl in noise_levels if get_result(noise_type, nl)]
        
        if metric_matrix.size > 0 and len(valid_levels_hm) > 0:
            im = ax.imshow(metric_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            ax.set_xticks(np.arange(len(valid_levels_hm)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels([f'{nl:.2f}' for nl in valid_levels_hm], fontsize=11)
            ax.set_yticklabels([m.capitalize() for m in metrics], fontsize=11)
            
            ax.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
            ax.set_title(f'Metrics Heatmap - {noise_type.capitalize()} Noise', fontsize=14, fontweight='bold')
            
            # Add text annotations
            for i in range(len(metrics)):
                for j in range(len(valid_levels_hm)):
                    text = ax.text(j, i, f'{metric_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'noise_heatmap.png'), dpi=300, bbox_inches='tight')
            print(f"Heatmap plot saved to {save_dir}/noise_heatmap.png")
            plt.close()
    
    # 6. Noise types comparison heatmap (if multiple types)
    if len(noise_types) > 1:
        # Compare F1 scores across types and levels
        fig, ax = plt.subplots(figsize=(12, 8))
        f1_matrix = np.array([[get_result(nt, nl)['f1'] for nl in noise_levels 
                              if get_result(nt, nl)] for nt in noise_types])
        valid_levels_hm = [nl for nl in noise_levels if any(get_result(nt, nl) for nt in noise_types)]
        
        if f1_matrix.size > 0:
            im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            ax.set_xticks(np.arange(len(valid_levels_hm)))
            ax.set_yticks(np.arange(len(noise_types)))
            ax.set_xticklabels([f'{nl:.2f}' for nl in valid_levels_hm], fontsize=11)
            ax.set_yticklabels([nt.capitalize() for nt in noise_types], fontsize=11)
            
            ax.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
            ax.set_ylabel('Noise Type', fontsize=12, fontweight='bold')
            ax.set_title('F1 Score Heatmap - All Noise Types', fontsize=14, fontweight='bold')
            
            for i in range(len(noise_types)):
                for j in range(len(valid_levels_hm)):
                    if get_result(noise_types[i], valid_levels_hm[j]):
                        val = get_result(noise_types[i], valid_levels_hm[j])['f1']
                        ax.text(j, i, f'{val:.3f}',
                               ha="center", va="center", color="black", fontsize=9, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F1 Score', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'noise_types_heatmap.png'), dpi=300, bbox_inches='tight')
            print(f"Noise types heatmap saved to {save_dir}/noise_types_heatmap.png")
            plt.close()
    
    has_history = any('history' in results[key] and results[key]['history'] for key in results.keys())
    if has_history:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_type = noise_types[0] if noise_types else 'replacement'
        for noise_level in noise_levels:
            result = get_result(plot_type, noise_level)
            if result and 'history' in result and result['history']:
                history = result['history']
                epochs = range(1, len(history['train_loss']) + 1)
                
                axes[0].plot(epochs, history['train_loss'], 'o-', linewidth=2, markersize=4,
                           label=f'Noise {noise_level:.2f}', alpha=0.8)
                
                axes[1].plot(epochs, history['val_accuracy'], 'o-', linewidth=2, markersize=4,
                           label=f'Noise {noise_level:.2f}', alpha=0.8)
                
                axes[2].plot(epochs, history['val_f1'], 'o-', linewidth=2, markersize=4,
                           label=f'Noise {noise_level:.2f}', alpha=0.8)
        
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        axes[2].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('F1 Score', fontsize=11)
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])
        
    
        for noise_level in noise_levels:
            result = get_result(plot_type, noise_level)
            if result and 'history' in result and result['history']:
                history = result['history']
                epochs = range(1, len(history['train_loss']) + 1)
                axes[3].plot(epochs, history['val_f1'], 'o-', linewidth=2, markersize=4,
                           label=f'Noise {noise_level:.2f}', alpha=0.8)
        
        axes[3].set_title('Validation F1 - All Noise Levels', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Epoch', fontsize=11)
        axes[3].set_ylabel('F1 Score', fontsize=11)
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noise_training_history.png'), dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_dir}/noise_training_history.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with Noise')
    parser.add_argument('--csv_path', type=str, default='/Users/victoriaportnaya/Downloads/dataset_33.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--json_path', type=str, default='/Users/victoriaportnaya/Downloads/formula_33.json',
                        help='Path to formula JSON file')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3],
                        help='Noise levels to test (fraction of symbols to corrupt)')
    parser.add_argument('--noise_types', type=str, nargs='+', 
                        default=['replacement'],
                        choices=['replacement', 'deletion', 'insertion', 'swap', 'shuffle', 'mixed'],
                        help='Types of noise to test')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.0, help='Classification threshold')
    parser.add_argument('--early_stop_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results_noise.json', help='Output JSON file')
    parser.add_argument('--plot_dir', type=str, default='plots_noise', help='Directory to save plots')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    sequences, labels, symbol_to_idx, omega_idx, vocab_size, formula_data, alphabet = load_data(
        args.csv_path, args.json_path
    )
    print(f"Loaded {len(sequences)} sequences")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Alphabet size: {vocab_size - 1}")
    
    print(f"\nTesting noise levels: {args.noise_levels}")
    print(f"Testing noise types: {args.noise_types}")
    print("="*60)
    
    all_results = {}
    
    total_experiments = len(args.noise_types) * len(args.noise_levels)
    experiment_num = 0
    
    print(f"Total experiments to run: {total_experiments}\n")
    
    for noise_type in args.noise_types:
        for noise_level in args.noise_levels:
            experiment_num += 1
            key = f"{noise_type}_{noise_level}"
            
            print(f"\n{'='*60}")
            print(f"[{experiment_num}/{total_experiments}] Training: {noise_type} (level={noise_level})")
            print(f"{'='*60}")
            
            test_metrics, history, model = train_model(
                sequences, labels, symbol_to_idx, omega_idx, alphabet, vocab_size,
                noise_level, noise_type, args.epochs, args.batch_size, args.lr, device, args.threshold,
                args.early_stop_patience
            )
            
            all_results[key] = {
                'noise_type': noise_type,
                'noise_level': noise_level,
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
                'history': history
            }
            
            print(f"\n✓ Completed [{experiment_num}/{total_experiments}] - Test Metrics:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    print("\nGenerating comparison plots...")
    plot_comparison(all_results, save_dir=args.plot_dir, compare_noise_types=len(args.noise_types) > 1)
    
    print("\n" + "="*80)
    print("SUMMARY - Noise Comparison")
    print("="*80)
    
    if len(args.noise_types) == 1:
        noise_type = args.noise_types[0]
        print(f"Noise Type: {noise_type}")
        print(f"{'Level':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*80)
        for key in sorted(all_results.keys(), key=lambda x: all_results[x]['noise_level']):
            r = all_results[key]
            if r['noise_type'] == noise_type:
                print(f"{r['noise_level']:<10.2f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    else:
        print(f"{'Type':<15} {'Level':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*80)
        for key in sorted(all_results.keys(), key=lambda x: (all_results[x]['noise_type'], all_results[x]['noise_level'])):
            r = all_results[key]
            print(f"{r['noise_type']:<15} {r['noise_level']:<10.2f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    print("="*80)
    
    baseline_key = None
    for key in all_results.keys():
        if all_results[key]['noise_level'] == 0.0:
            baseline_key = key
            break
    
    if baseline_key:
        baseline = all_results[baseline_key]
        print(f"\nPerformance Degradation from Baseline (noise=0.0):")
        print(f"{'Type':<15} {'Level':<10} {'Accuracy Δ':<15} {'F1 Δ':<15}")
        print("-"*60)
        for key in sorted(all_results.keys(), key=lambda x: (all_results[x]['noise_type'], all_results[x]['noise_level'])):
            r = all_results[key]
            if r['noise_level'] > 0:
                acc_delta = r['accuracy'] - baseline['accuracy']
                f1_delta = r['f1'] - baseline['f1']
                print(f"{r['noise_type']:<15} {r['noise_level']:<10.2f} {acc_delta:>+12.4f} {f1_delta:>+12.4f}")


if __name__ == '__main__':
    main()
