"""
Generate a more complicated dataset with:
- More complex formula (more disjunctions and conjunctions)
- Longer sequences
- 10,000 sequences verified with evaluator
"""

from ltl_formulas.formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator,
    save_formula_json,
    save_dataset_csv
)
import os

# Parameters for more complex formula
formula_id = 34  # Different ID to distinguish from formula 33
alphabet_size = 50  # Larger alphabet
num_disjunctions = 4  # More disjunctions
num_conjunctions = 3  # More conjunctions per disjunction
sequence_length = 10  # Longer sequences
num_positive = 5000
num_negative = 5000
total_sequences = num_positive + num_negative
seed = 42

# Create output directory
output_dir = "generator_updated/dataset"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print(f"Generating Complex Dataset - Formula {formula_id}")
print("=" * 70)
print(f"Parameters:")
print(f"  Alphabet size: {alphabet_size}")
print(f"  Disjunctions: {num_disjunctions}")
print(f"  Conjunctions per disjunction: {num_conjunctions}")
print(f"  Sequence length: {sequence_length} (longer)")
print(f"  Total sequences: {total_sequences}")
print(f"  Positive sequences: {num_positive}")
print(f"  Negative sequences: {num_negative}")
print(f"  Seed: {seed}")
print("=" * 70)

# Generate formula
print("\nGenerating complex formula...")
formula_gen = FormulaGenerator(
    alphabet_size=alphabet_size,
    num_disjunctions=num_disjunctions,
    num_conjunctions=num_conjunctions,
    seed=seed
)

formula = formula_gen.generate_formula()
formula_str = formula_gen.formula_to_string(formula)
print(f"Formula: {formula_str}")
print(f"\nFormula complexity:")
print(f"  - {num_disjunctions} disjunction clauses")
print(f"  - {num_conjunctions} conjunctions per clause")
print(f"  - Total subformulas: {num_disjunctions * num_conjunctions}")

# Generate synthetic data
print(f"\nGenerating {total_sequences} sequences (this may take a while)...")
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=seed
)

# Generate dataset with verification
print("Generating sequences and verifying labels...")
verified_dataset = []
evaluator = FormulaEvaluator()

positive_generated = 0
negative_generated = 0
positive_incorrect = 0
negative_incorrect = 0

# Generate positive sequences
print(f"\nGenerating {num_positive} positive sequences...")
while positive_generated < num_positive:
    seq = data_gen.generate_sequence_satisfying_formula(formula, sequence_length)
    actual_satisfies = evaluator.evaluate_formula(seq, formula)
    
    if actual_satisfies:
        verified_dataset.append((seq, 1))
        positive_generated += 1
        if positive_generated % 500 == 0:
            print(f"  Generated {positive_generated}/{num_positive} positive sequences...")
    else:
        positive_incorrect += 1

# Generate negative sequences
print(f"\nGenerating {num_negative} negative sequences...")
while negative_generated < num_negative:
    seq = data_gen.generate_sequence_not_satisfying_formula(formula, sequence_length)
    actual_satisfies = evaluator.evaluate_formula(seq, formula)
    
    if not actual_satisfies:
        verified_dataset.append((seq, 0))
        negative_generated += 1
        if negative_generated % 500 == 0:
            print(f"  Generated {negative_generated}/{num_negative} negative sequences...")
    else:
        negative_incorrect += 1

print(f"\n✓ Generated {len(verified_dataset)} verified sequences")
print(f"  Positive: {positive_generated}")
print(f"  Negative: {negative_generated}")
print(f"  Incorrect labels filtered: {positive_incorrect + negative_incorrect}")

# Final verification pass
print("\nPerforming final verification pass...")
correct = 0
incorrect = 0

for seq, label in verified_dataset:
    actual_satisfies = evaluator.evaluate_formula(seq, formula)
    expected_satisfies = bool(label)
    
    if actual_satisfies == expected_satisfies:
        correct += 1
    else:
        incorrect += 1

accuracy = correct / len(verified_dataset) * 100 if len(verified_dataset) > 0 else 0

print(f"Final verification:")
print(f"  Correct labels: {correct}/{len(verified_dataset)} ({accuracy:.2f}%)")
print(f"  Incorrect labels: {incorrect}")

if incorrect == 0:
    print("✓ All sequences verified correctly!")
else:
    print(f"⚠ Warning: Found {incorrect} sequences with incorrect labels!")

# Save formula
formula_filename = os.path.join(output_dir, f"formula_{formula_id}.json")
save_formula_json(
    formula, 
    formula_filename,
    metadata={
        'formula_id': formula_id,
        'alphabet_size': alphabet_size,
        'num_disjunctions': num_disjunctions,
        'num_conjunctions': num_conjunctions,
        'sequence_length': sequence_length,
        'num_positive': num_positive,
        'num_negative': num_negative,
        'total_sequences': len(verified_dataset),
        'seed': seed,
        'alphabet': formula_gen.alphabet,
        'verification_accuracy': accuracy,
        'incorrect_labels': incorrect,
        'complexity': 'high'
    }
)
print(f"\n✓ Saved formula to: {formula_filename}")

# Save dataset
dataset_filename = os.path.join(output_dir, f"dataset_{formula_id}.csv")
save_dataset_csv(verified_dataset, dataset_filename)
print(f"✓ Saved dataset to: {dataset_filename}")

print(f"\n{'='*70}")
print("Dataset generation and verification complete!")
print(f"{'='*70}")
print(f"\nDataset summary:")
print(f"  Total sequences: {len(verified_dataset)}")
print(f"  Positive: {positive_generated}")
print(f"  Negative: {negative_generated}")
print(f"  Verification accuracy: {accuracy:.2f}%")
print(f"  Formula complexity: {num_disjunctions} disjunctions, {num_conjunctions} conjunctions each")
print(f"  Sequence length: {sequence_length}")
print(f"  Formula file: {formula_filename}")
print(f"  Dataset file: {dataset_filename}")

