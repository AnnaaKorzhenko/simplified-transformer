"""
Example showing how to use custom multi-character alphabet symbols
like 'p1', 'p2', 'pi'
"""

from formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator,
    save_dataset_csv
)

# Example: Using custom alphabet symbols
print("=" * 70)
print("Example: Custom Multi-Character Alphabet Symbols")
print("=" * 70)

# Option 1: Specify custom alphabet
custom_alphabet = ['p1', 'p2', 'p3', 'p4', 'pi']

formula_gen = FormulaGenerator(
    alphabet_size=5,
    num_disjunctions=2,
    num_conjunctions=2,
    alphabet=custom_alphabet,  # Use custom symbols
    seed=123
)

print(f"\nAlphabet: {formula_gen.alphabet}")
formula = formula_gen.generate_formula()
print(f"\nGenerated Formula:\n{formula_gen.formula_to_string(formula)}")

# Generate data
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=123
)

dataset = data_gen.generate_dataset(
    formula=formula,
    sequence_length=10,
    num_positive=30,
    num_negative=30,
    seed=123
)

print(f"\nDataset: {len(dataset)} sequences")
print("\nSample sequences:")
for i, (seq, label) in enumerate(dataset[:5]):
    seq_str = ','.join(seq)
    satisfies = FormulaEvaluator.evaluate_formula(seq, formula)
    print(f"  {i+1}. [{seq_str}] -> label={label}, satisfies={satisfies}")

# Save to CSV
save_dataset_csv(dataset, "custom_alphabet_dataset.csv")
print("\nSaved dataset to custom_alphabet_dataset.csv")

# Option 2: Default behavior (automatically generates p1, p2, p3, ...)
print("\n" + "=" * 70)
print("Example: Default Alphabet (auto-generated p1, p2, p3, ...)")
print("=" * 70)

formula_gen2 = FormulaGenerator(
    alphabet_size=4,  # Will generate ['p1', 'p2', 'p3', 'p4']
    num_disjunctions=2,
    num_conjunctions=2,
    seed=456
)

print(f"\nAlphabet: {formula_gen2.alphabet}")
formula2 = formula_gen2.generate_formula()
print(f"\nGenerated Formula:\n{formula_gen2.formula_to_string(formula2)}")

# Option 3: Mixed symbols
print("\n" + "=" * 70)
print("Example: Mixed Symbol Names")
print("=" * 70)

mixed_alphabet = ['p1', 'p2', 'pi', 'alpha', 'beta']

formula_gen3 = FormulaGenerator(
    alphabet_size=5,
    num_disjunctions=2,
    num_conjunctions=2,
    alphabet=mixed_alphabet,
    seed=789
)

print(f"\nAlphabet: {formula_gen3.alphabet}")
formula3 = formula_gen3.generate_formula()
print(f"\nGenerated Formula:\n{formula_gen3.formula_to_string(formula3)}")

# Generate a sample sequence to show it works
data_gen3 = SyntheticDataGenerator(
    alphabet=formula_gen3.alphabet,
    seed=789
)

sample_seq = data_gen3.generate_random_sequence(8)
print(f"\nSample sequence: {sample_seq}")
print(f"  (as string: {','.join(sample_seq)})")

