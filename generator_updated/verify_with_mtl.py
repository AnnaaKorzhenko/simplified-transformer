"""
Verify generated sequences against formulas using py-metric-temporal-logic library.

This script converts our temporal logic formulas to MTL format and uses the
py-metric-temporal-logic library to verify if sequences satisfy the formulas.
"""

import os
import json
import csv
from typing import List, Tuple
from ltl_formulas.formula_generator import load_formula_json
import mtl


def convert_formula_to_mtl(formula: List[List[Tuple[str, str, str]]]) -> str:
    """
    Convert our formula format to MTL format.
    
    Our formula: Disjunction of conjunctions of "not a until b and c"
    MTL format: We can use the parse API or Python operators
    
    "not a until b and c" = (¬a U (b ∧ c))
    In MTL: (~a U (b & c))
    
    Note: MTL parser requires the entire formula to be wrapped in parentheses
    """
    
    # Build MTL formula string
    disjunction_parts = []
    
    for conj_clause in formula:
        # Build conjunction: all subformulas must be true
        conjunction_parts = []
        
        for a, b, c in conj_clause:
            # "not a until b and c" = (~a U (b & c))
            until_formula = f"(~{a} U ({b} & {c}))"
            conjunction_parts.append(until_formula)
        
        # Join conjunctions with &
        conjunction_str = " & ".join(conjunction_parts)
        disjunction_parts.append(f"({conjunction_str})")
    
    # Join disjunctions with |
    # Wrap the entire formula in parentheses for the parser
    if len(disjunction_parts) == 1:
        mtl_formula_str = disjunction_parts[0]
    else:
        mtl_formula_str = f"({' | '.join(disjunction_parts)})"
    
    return mtl_formula_str


def sequence_to_mtl_data(sequence: List[str], alphabet: List[str], seen_so_far: bool = True) -> dict:
    """
    Convert a sequence to MTL data format.
    
    MTL expects: dict mapping atomic predicate names to lists of (time, val) pairs
    For discrete sequences, we use integer time steps (0, 1, 2, ...)
    
    Args:
        sequence: List of symbols (one per position)
        alphabet: List of all alphabet symbols
        seen_so_far: If True, use "seen so far" semantics where symbols stay True
                    after first appearance. This allows MTL (b & c) to work correctly
                    when both symbols have appeared by a certain time point.
                    If False, use point-in-time semantics (only True at exact position).
    
    We need to provide explicit True/False values for each symbol at each time step
    to avoid evaluation errors with the MTL library.
    """
    data = {}
    max_time = len(sequence) - 1
    
    # Track which symbols have been seen
    seen_symbols = set()
    
    # For each symbol in the alphabet, provide data points
    for symbol in alphabet:
        data[symbol] = []
        
        # Build the time series for this symbol
        for time in range(len(sequence)):
            current_symbol = sequence[time]
            
            if seen_so_far:
                # "Seen so far" semantics: symbol becomes True when it appears and stays True
                if current_symbol == symbol:
                    seen_symbols.add(symbol)
                data[symbol].append((time, symbol in seen_symbols))
            else:
                # Point-in-time semantics: only True at exact position
                data[symbol].append((time, current_symbol == symbol))
    
    return data


def verify_sequence_with_mtl(sequence: List[str], 
                             formula_str: str,
                             alphabet: List[str]) -> bool:
    """
    Verify a sequence using MTL library.
    
    Args:
        sequence: List of symbols (e.g., ['p1', 'p2', 'p3'])
        formula_str: MTL formula string
        alphabet: List of all alphabet symbols
    
    Returns:
        True if sequence satisfies formula, False otherwise
    """
    try:
        # Parse the formula
        phi = mtl.parse(formula_str)
        
        # Convert sequence to MTL data format
        data = sequence_to_mtl_data(sequence, alphabet)
        
        # Evaluate with boolean semantics (quantitative=False)
        result = phi(data, quantitative=False)
        
        return bool(result)
    except Exception as e:
        print(f"Error evaluating sequence {sequence}: {e}")
        return False


def verify_full_dataset(formula_file: str, dataset_file: str) -> dict:
    """
    Verify all sequences in dataset using MTL library.
    """
    # Load formula
    formula, metadata = load_formula_json(formula_file)
    alphabet = metadata.get('alphabet', [])
    
    # Convert formula to MTL format
    mtl_formula_str = convert_formula_to_mtl(formula)
    print(f"MTL Formula: {mtl_formula_str}\n")
    
    # Parse the formula once
    try:
        phi = mtl.parse(mtl_formula_str)
        print("✓ MTL formula parsed successfully\n")
    except Exception as e:
        print(f"✗ Error parsing MTL formula: {e}")
        return {'error': str(e)}
    
    # Load full dataset
    dataset = []
    with open(dataset_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = row['sequence'].split(',')
            label = int(row['label'])
            dataset.append((sequence, label))
    
    print(f"Verifying {len(dataset)} sequences using py-metric-temporal-logic...\n")
    
    # Verify using MTL library
    correct = 0
    incorrect = 0
    errors = []
    
    for i, (seq, expected_label) in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Verified {i + 1}/{len(dataset)} sequences...")
        
        try:
            # Convert sequence to MTL data format
            data = sequence_to_mtl_data(seq, alphabet)
            
            # Evaluate with boolean semantics
            actual_satisfies = phi(data, quantitative=False)
            expected_satisfies = bool(expected_label)
            
            if actual_satisfies == expected_satisfies:
                correct += 1
            else:
                incorrect += 1
                if len(errors) < 10:  # Keep first 10 errors
                    errors.append({
                        'sequence': seq,
                        'expected': expected_label,
                        'actual': 1 if actual_satisfies else 0
                    })
        except Exception as e:
            if len(errors) < 10:
                errors.append({
                    'sequence': seq,
                    'error': str(e)
                })
    
    accuracy = correct / len(dataset) * 100 if len(dataset) > 0 else 0
    
    return {
        'total_checked': len(dataset),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'errors': errors[:10],  # Return first 10 errors
        'mtl_formula': mtl_formula_str,
        'verification_method': 'py-metric-temporal-logic (MTL library)'
    }


def main():
    """Main function to verify the large single dataset using MTL."""
    
    formula_file = "dataset/formula_33.json"
    dataset_file = "dataset/dataset_33.csv"
    
    if not os.path.exists(formula_file):
        print(f"Error: Formula file not found: {formula_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        return
    
    print("=" * 70)
    print("Verifying Sequences with py-metric-temporal-logic (MTL)")
    print("=" * 70)
    print(f"Formula file: {formula_file}")
    print(f"Dataset file: {dataset_file}")
    print("=" * 70)
    
    # Load formula to show it
    formula, metadata = load_formula_json(formula_file)
    from ltl_formulas.formula_generator import FormulaGenerator
    gen = FormulaGenerator(1, 1, 1)  # Dummy, just for string conversion
    gen.alphabet = metadata.get('alphabet', [])
    formula_str = gen.formula_to_string(formula)
    print(f"\nOriginal Formula: {formula_str}")
    
    # Verify full dataset
    results = verify_full_dataset(formula_file, dataset_file)
    
    if 'error' in results:
        print(f"\nError: {results['error']}")
        return
    
    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)
    print(f"Verification method: {results['verification_method']}")
    print(f"MTL Formula: {results['mtl_formula']}")
    print(f"Total sequences checked: {results['total_checked']}")
    print(f"Correct labels: {results['correct']}")
    print(f"Incorrect labels: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    if results['errors']:
        print(f"\nFirst {len(results['errors'])} errors:")
        for i, error in enumerate(results['errors'], 1):
            if 'error' in error:
                print(f"  {i}. Sequence {error['sequence']}: {error['error']}")
            else:
                print(f"  {i}. Sequence {error['sequence']}: expected {error['expected']}, got {error['actual']}")
    
    # Save results
    results_file = "mtl_verification_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

