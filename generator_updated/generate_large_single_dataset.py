"""
Generate a single dataset with 10,000 sequences.
Formula 33, alphabet size 33, 2 disjunctions, 2 conjunctions, sequence length 5
"""

from ltl_formulas.formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator,
    save_formula_json,
    save_dataset_csv
)
import os

# Parameters
formula_id = 33
alphabet_size = 33
num_disjunctions = 2
num_conjunctions = 2
sequence_length = 5
num_positive = 5000
num_negative = 5000
seed = 42

# Create output directory
output_dir = "large_single_dataset"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "formulas"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)

print("=" * 70)
print(f"Generating Large Single Dataset - Formula {formula_id}")
print("=" * 70)
print(f"Parameters:")
print(f"  Alphabet size: {alphabet_size}")
print(f"  Disjunctions: {num_disjunctions}")
print(f"  Conjunctions per disjunction: {num_conjunctions}")
print(f"  Sequence length: {sequence_length}")
print(f"  Positive sequences: {num_positive}")
print(f"  Negative sequences: {num_negative}")
print(f"  Total sequences: {num_positive + num_negative}")
print(f"  Seed: {seed}")
print("=" * 70)

# Generate formula
formula_gen = FormulaGenerator(
    alphabet_size=alphabet_size,
    num_disjunctions=num_disjunctions,
    num_conjunctions=num_conjunctions,
    seed=seed
)

formula = formula_gen.generate_formula()
formula_str = formula_gen.formula_to_string(formula)

print(f"\nGenerated Formula:")
print(f"  {formula_str}")

# Generate synthetic data
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=seed
)

print(f"\nGenerating dataset (this may take a while for 10,000 sequences)...")
dataset = data_gen.generate_dataset(
    formula=formula,
    sequence_length=sequence_length,
    num_positive=num_positive,
    num_negative=num_negative,
    seed=seed
)

# Verify correctness
evaluator = FormulaEvaluator()
correct = 0
for seq, label in dataset:
    satisfies = evaluator.evaluate_formula(seq, formula)
    if satisfies == bool(label):
        correct += 1

accuracy = 100 * correct / len(dataset)
print(f"\nDataset generated: {len(dataset)} sequences")
print(f"Label accuracy: {correct}/{len(dataset)} ({accuracy:.1f}%)")

# Show sample sequences
print(f"\nSample sequences:")
for i, (seq, label) in enumerate(dataset[:5]):
    print(f"  {i+1}. {seq} -> {label}")

# Save formula
formula_filename = os.path.join(output_dir, "formulas", f"formula_{formula_id}.json")
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
        'seed': seed,
        'alphabet': formula_gen.alphabet
    }
)
print(f"\nSaved formula to: {formula_filename}")

# Save dataset
dataset_filename = os.path.join(output_dir, "datasets", f"dataset_{formula_id}.csv")
save_dataset_csv(dataset, dataset_filename)
print(f"Saved dataset to: {dataset_filename}")

# Save summary
summary = {
    'parameters': {
        'formula_id': formula_id,
        'alphabet_size': alphabet_size,
        'num_disjunctions': num_disjunctions,
        'num_conjunctions': num_conjunctions,
        'sequence_length': sequence_length,
        'num_positive': num_positive,
        'num_negative': num_negative,
        'seed': seed
    },
    'formula': formula_str,
    'alphabet': formula_gen.alphabet,
    'total_sequences': len(dataset),
    'positive_count': sum(1 for _, label in dataset if label == 1),
    'negative_count': sum(1 for _, label in dataset if label == 0),
    'label_accuracy': accuracy,
    'formula_file': formula_filename,
    'dataset_file': dataset_filename
}

summary_filename = os.path.join(output_dir, "summary.json")
with open(summary_filename, 'w') as f:
    import json
    json.dump(summary, f, indent=2)
print(f"Saved summary to: {summary_filename}")

print(f"\n{'='*70}")
print("Generation complete!")
print(f"{'='*70}")
print(f"\nAll files saved in directory: {output_dir}/")
print(f"  Formula: {output_dir}/formulas/formula_{formula_id}.json")
print(f"  Dataset: {output_dir}/datasets/dataset_{formula_id}.csv")
print(f"  Summary: {output_dir}/summary.json")




