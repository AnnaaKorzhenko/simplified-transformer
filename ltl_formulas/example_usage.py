"""
Example usage of the formula generator and synthetic data generator
"""

from formula_generator import (
    FormulaGenerator,
    FormulaEvaluator,
    SyntheticDataGenerator,
    save_dataset_csv,
    save_formula_json,
    generate_multiple_formulas_and_datasets
)

# Example 1: Generate a single formula and dataset
print("=" * 70)
print("Example 1: Single Formula and Dataset")
print("=" * 70)

formula_gen = FormulaGenerator(
    alphabet_size=4,      # Size of alphabet (pi)
    num_disjunctions=2,   # Number of OR clauses
    num_conjunctions=3,   # Number of AND clauses per OR
    seed=123
)

formula = formula_gen.generate_formula()
print(f"\nGenerated Formula:\n{formula_gen.formula_to_string(formula)}")

# Generate synthetic data
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=123
)

dataset = data_gen.generate_dataset(
    formula=formula,
    sequence_length=15,    # Length of each sequence
    num_positive=100,      # Number of sequences that satisfy formula
    num_negative=100,      # Number of sequences that don't satisfy
    seed=123
)

# Verify correctness
correct = sum(
    1 for seq, label in dataset
    if FormulaEvaluator.evaluate_formula(seq, formula) == bool(label)
)
print(f"\nCorrect labels: {correct}/{len(dataset)} ({100*correct/len(dataset):.1f}%)")

# Save results
save_formula_json(formula, "formula.json", {
    "alphabet_size": 4,
    "num_disjunctions": 2,
    "num_conjunctions": 3
})
save_dataset_csv(dataset, "dataset.csv")
print("\nSaved formula to formula.json and dataset to dataset.csv")


# Example 2: Generate multiple formulas and datasets
print("\n" + "=" * 70)
print("Example 2: Multiple Formulas and Datasets")
print("=" * 70)

results = generate_multiple_formulas_and_datasets(
    alphabet_size=3,
    num_disjunctions=2,
    num_conjunctions=2,
    sequence_length=10,
    num_positive=50,
    num_negative=50,
    num_formulas=3,
    seed=456
)

print(f"\nGenerated {len(results)} formulas with their datasets")


# Example 3: Custom parameters
print("\n" + "=" * 70)
print("Example 3: Custom Parameters")
print("=" * 70)

custom_formula_gen = FormulaGenerator(
    alphabet_size=6,      # Larger alphabet
    num_disjunctions=3,   # More disjunctions
    num_conjunctions=2,   # Fewer conjunctions per disjunction
    seed=789
)

custom_formula = custom_formula_gen.generate_formula()
print(f"\nCustom Formula:\n{custom_formula_gen.formula_to_string(custom_formula)}")

custom_data_gen = SyntheticDataGenerator(
    alphabet=custom_formula_gen.alphabet,
    seed=789
)

custom_dataset = custom_data_gen.generate_dataset(
    formula=custom_formula,
    sequence_length=20,   # Longer sequences
    num_positive=200,     # More positive examples
    num_negative=200,     # More negative examples
    seed=789
)

positive_count = sum(1 for _, label in custom_dataset if label == 1)
negative_count = sum(1 for _, label in custom_dataset if label == 0)
print(f"\nDataset: {len(custom_dataset)} sequences ({positive_count} positive, {negative_count} negative)")

