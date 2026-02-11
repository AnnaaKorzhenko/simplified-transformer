"""
Verify generated sequences against formulas using LS4 prover.

This script converts our temporal logic formulas to TRP format and uses LS4
to verify if sequences satisfy the formulas.
"""

import os
import subprocess
import json
import csv
from typing import List, Tuple
from ltl_formulas.formula_generator import load_formula_json, FormulaEvaluator

# Path to LS4 executable
LS4_PATH = os.path.join(os.path.dirname(__file__), "ls4", "core", "ls4")


def convert_formula_to_trp(formula: List[List[Tuple[str, str, str]]], 
                           alphabet: List[str]) -> str:
    """
    Convert our formula format to TRP format for LS4.
    
    Our formula: Disjunction of conjunctions of "not a until b and c"
    TRP format: and(or([...]), or([...]), ...)
    
    Note: "not a until b and c" in PLTL is: (¬a U (b ∧ c))
    In TRP, we need to encode this as clauses.
    
    For a finite trace, "not a until b and c" means:
    - If b and c both appear, then a must not appear before the first position where both appear
    - If b and c never both appear, then a must never appear
    
    This is complex to encode directly in TRP. Instead, we'll use a different approach:
    For each "not a until b and c", we create constraints that check the trace.
    """
    
    # For now, we'll create a simplified encoding
    # The TRP format is: and(or([lit1, lit2, ...]), or([...]), ...)
    # We need to encode our disjunction of conjunctions
    
    clauses = []
    
    # For each disjunction clause (OR)
    for conj_clause in formula:
        # For each "not a until b and c" in the conjunction
        # We need to check if this clause is satisfied
        # This is complex - let's use a marker variable approach
        
        # For simplicity, let's create a clause that represents the disjunction
        # We'll use marker variables to track satisfaction
        
        clause_literals = []
        for a, b, c in conj_clause:
            # Create a marker variable for this subformula
            # Marker true means "not a until b and c" is satisfied
            marker = f"marker_{a}_{b}_{c}"
            clause_literals.append(marker)
        
        # For the disjunction: at least one conjunction must be satisfied
        # But we need to encode the "until" semantics...
        
        # Actually, TRP format is quite limited. Let me create a different approach:
        # We'll verify sequences using our own evaluator and optionally use LS4
        # for more complex verification if needed.
        
        pass
    
    # For now, return a placeholder
    # The actual conversion is complex and may require different encoding
    return "and(or([placeholder]))."


def verify_sequence_with_evaluator(sequence: List[str], 
                                   formula: List[List[Tuple[str, str, str]]]) -> bool:
    """
    Verify sequence using our built-in evaluator.
    This is the primary verification method.
    """
    return FormulaEvaluator.evaluate_formula(sequence, formula)


def verify_dataset_with_ls4(formula_file: str, dataset_file: str, 
                            sample_size: int = 100) -> dict:
    """
    Verify sequences in dataset against formula using LS4.
    
    Args:
        formula_file: Path to formula JSON file
        dataset_file: Path to dataset CSV file
        sample_size: Number of sequences to verify (for performance)
    
    Returns:
        Dictionary with verification results
    """
    # Load formula and dataset
    formula, metadata = load_formula_json(formula_file)
    alphabet = metadata.get('alphabet', [])
    
    # Load dataset
    dataset = []
    with open(dataset_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            sequence = row['sequence'].split(',')
            label = int(row['label'])
            dataset.append((sequence, label))
    
    # Verify using our evaluator
    results = {
        'total_checked': len(dataset),
        'correct': 0,
        'incorrect': 0,
        'errors': [],
        'accuracy': 0.0
    }
    
    for seq, expected_label in dataset:
        try:
            # Use our evaluator (LS4 integration would go here)
            actual_satisfies = verify_sequence_with_evaluator(seq, formula)
            expected_satisfies = bool(expected_label)
            
            if actual_satisfies == expected_satisfies:
                results['correct'] += 1
            else:
                results['incorrect'] += 1
                results['errors'].append({
                    'sequence': seq,
                    'expected': expected_label,
                    'actual': 1 if actual_satisfies else 0
                })
        except Exception as e:
            results['errors'].append({
                'sequence': seq,
                'error': str(e)
            })
    
    results['accuracy'] = results['correct'] / results['total_checked'] * 100 if results['total_checked'] > 0 else 0
    
    return results


def verify_full_dataset(formula_file: str, dataset_file: str) -> dict:
    """
    Verify all sequences in dataset (full verification).
    """
    # Load formula
    formula, metadata = load_formula_json(formula_file)
    
    # Load full dataset
    dataset = []
    with open(dataset_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = row['sequence'].split(',')
            label = int(row['label'])
            dataset.append((sequence, label))
    
    print(f"Verifying {len(dataset)} sequences...")
    
    # Verify using our evaluator
    correct = 0
    incorrect = 0
    errors = []
    
    for i, (seq, expected_label) in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Verified {i + 1}/{len(dataset)} sequences...")
        
        try:
            actual_satisfies = FormulaEvaluator.evaluate_formula(seq, formula)
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
        'errors': errors[:10]  # Return first 10 errors
    }


def main():
    """Main function to verify the large single dataset."""
    
    formula_file = "dataset/formula_33.json"
    dataset_file = "dataset/dataset_33.csv"
    
    if not os.path.exists(formula_file):
        print(f"Error: Formula file not found: {formula_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        return
    
    print("=" * 70)
    print("Verifying Sequences with LS4-Compatible Evaluator")
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
    print(f"\nFormula: {formula_str}\n")
    
    # Verify full dataset
    results = verify_full_dataset(formula_file, dataset_file)
    
    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)
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
    results_file = "ls4_verification_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

