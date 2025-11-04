"""
Shared utilities for training scripts.
"""

import os
import json
import numpy as np
import random
from typing import List, Tuple

from ltl_formulas.formula_generator import load_dataset_csv, load_formula_json


def extract_features(sequence: List[str], alphabet: List[str], sequence_length: int) -> np.ndarray:
    """Extract features from a sequence for logistic regression."""
    features = []
    
    # One-hot encoding for each position
    for i in range(sequence_length):
        position_features = [0.0] * len(alphabet)
        if i < len(sequence):
            if sequence[i] in alphabet:
                idx = alphabet.index(sequence[i])
                position_features[idx] = 1.0
        features.extend(position_features)
    
    # Frequency counts
    freq_features = [0.0] * len(alphabet)
    for sym in sequence:
        if sym in alphabet:
            idx = alphabet.index(sym)
            freq_features[idx] += 1.0
    features.extend(freq_features)
    
    # First occurrence positions (normalized)
    first_positions = [sequence_length] * len(alphabet)
    for i, sym in enumerate(sequence):
        if sym in alphabet:
            idx = alphabet.index(sym)
            if first_positions[idx] == sequence_length:
                first_positions[idx] = i
    first_positions = [pos / sequence_length for pos in first_positions]
    features.extend(first_positions)
    
    # Last occurrence positions (normalized)
    last_positions = [-1] * len(alphabet)
    for i, sym in enumerate(sequence):
        if sym in alphabet:
            idx = alphabet.index(sym)
            last_positions[idx] = i
    last_positions = [pos / sequence_length if pos >= 0 else 0.0 for pos in last_positions]
    features.extend(last_positions)
    
    return np.array(features)


def split_dataset(dataset: List[Tuple[List[str], int]], test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    np.random.seed(random_state)
    random.seed(random_state)
    
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - test_size))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    
    X_train = [seq for seq, _ in train_data]
    y_train = [label for _, label in train_data]
    X_test = [seq for seq, _ in test_data]
    y_test = [label for _, label in test_data]
    
    return X_train, y_train, X_test, y_test


def load_single_dataset(dataset_dir: str, formula_id: int = 33):
    """Load a single formula and dataset."""
    # Resolve path relative to project root
    if not os.path.isabs(dataset_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(project_root, dataset_dir)
    
    # Try different possible path structures
    possible_formula_paths = [
        os.path.join(dataset_dir, "formulas", f"formula_{formula_id}.json"),
        os.path.join(dataset_dir, f"formula_{formula_id}.json"),
    ]
    
    possible_dataset_paths = [
        os.path.join(dataset_dir, "datasets", f"dataset_{formula_id}.csv"),
        os.path.join(dataset_dir, f"dataset_{formula_id}.csv"),
    ]
    
    formula_path = None
    for path in possible_formula_paths:
        if os.path.exists(path):
            formula_path = path
            break
    
    if formula_path is None:
        raise FileNotFoundError(f"Formula file not found. Tried: {possible_formula_paths}")
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        raise FileNotFoundError(f"Dataset file not found. Tried: {possible_dataset_paths}")
    
    formula, metadata = load_formula_json(formula_path)
    alphabet = metadata.get('alphabet', [f'p{i+1}' for i in range(33)])
    dataset = load_dataset_csv(dataset_path)
    sequence_length = metadata.get('sequence_length', 5)
    
    return formula, dataset, alphabet, sequence_length



