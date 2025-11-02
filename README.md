# LTL Formula and Synthetic Data Generator

This code generates LTL formulas in the form of disjunctions of conjunctions where each conjunction contains formulas of the form "not a until b and c" (where a, b, c are alphabet symbols). It then generates synthetic sequence data labeled according to whether they satisfy these formulas.

## Formula Structure

The formulas generated have the following structure:
- **Disjunction (OR)** of multiple clauses
- Each clause is a **Conjunction (AND)** of multiple subformulas
- Each subformula is: **"not a until b and c"** 

Example of valid formula:
```
((¬a U (b ∧ c)) ∧ (¬d U (e ∧ f))) ∨ ((¬g U (h ∧ i)) ∧ (¬j U (k ∧ l)))
```

## Parameters

### Formula Generation:
- **`alphabet_size π`**: Size of the alphabet
- **`alphabet Σ`** : Custom alphabet (e.g., `['a', 'b', 'c', 'd']`)
  - If not provided, automatically generates symbols like `'p1'`, `'p2'`, `'pi'`
- **`num_disjunctions`**: Number of OR clauses in the formula
- **`num_conjunctions`**: Number of AND clauses per OR clause

### Dataset Generation:
- **`sequence_length`**: Length of each generated sequence (trace)
- **`num_positive`**: Number of sequences that satisfy the formula (label = 1)
- **`num_negative`**: Number of sequences that do not satisfy the formula (label = 0)

## Usage

```python
from formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator
)

# Step 1: Generate a formula
formula_gen = FormulaGenerator(
    alphabet_size=5,        # Alphabet size (π)
    num_disjunctions=2,     # Number of OR clauses
    num_conjunctions=2,     # Number of AND clauses per OR
    seed=42                 # For reproducibility
)
# By default generates alphabet ['p1', 'p2', 'p3', 'p4', 'p5']

formula = formula_gen.generate_formula()
print(formula_gen.formula_to_string(formula))

# Step 2: Generate synthetic data
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=42
)

dataset = data_gen.generate_dataset(
    formula=formula,
    sequence_length=10,     # Length of sequences
    num_positive=50,        # Number of positive examples
    num_negative=50,        # Number of negative examples
    seed=42
)

# Step 3: Now we have the dataset!
# dataset is a list of (sequence, label) tuples
# sequence is a list of symbols (strings)
# label = 1 if sequence satisfies formula, 0 otherwise
for sequence, label in dataset[:5]:
    print(f"{sequence} -> {label}")
```

### If you have your own alphabet

```python
# Use custom multi-character symbols
custom_alphabet = ['p1', 'p2', 'p3', 'pi', 'alpha']

formula_gen = FormulaGenerator(
    alphabet_size=5,
    num_disjunctions=2,
    num_conjunctions=2,
    alphabet=custom_alphabet,  # Custom symbols
    seed=42
)

formula = formula_gen.generate_formula()
# Formula will use symbols like 'p1', 'p2', 'pi', etc.

### Evaluate a Sequence

```python
# Check if a sequence satisfies a formula
# Sequences are lists of symbols 
sequence = ['p1', 'p2', 'p3', 'p1', 'p2']
satisfies = FormulaEvaluator.evaluate_formula(sequence, formula)
print(f"Sequence {sequence} satisfies formula: {satisfies}")
```

### Save and Load Data

```python
from formula_generator import save_dataset_csv, save_formula_json, load_dataset_csv, load_formula_json

# Save dataset to CSV
save_dataset_csv(dataset, "my_dataset.csv")

# Save formula to JSON
save_formula_json(formula, "my_formula.json", metadata={
    "alphabet_size": 5,
    "num_disjunctions": 2,
    "num_conjunctions": 2
})

### Generate Multiple Formulas

```python
from formula_generator import generate_multiple_formulas_and_datasets

results = generate_multiple_formulas_and_datasets(
    alphabet_size=4,
    num_disjunctions=2,
    num_conjunctions=2,
    sequence_length=10,
    num_positive=50,
    num_negative=50,
    num_formulas=5,         # Generate 5 different formulas
    seed=123
)

# results is a list of (formula, dataset) tuples
for i, (formula, dataset) in enumerate(results):
    print(f"Formula {i+1}: {len(dataset)} sequences")
```

## Examples

Run the main example:
```bash
python3 formula_generator.py
```

Run additional usage examples:
```bash
python3 example_usage.py
```

## Formula Evaluation Procedure

For a formula "not a until b and c":
This means: *a does not appear in the sequence until both b and c have appeared*
That is, if a appears at position i, then both b and c must have appeared before position i. If b and c never both appear, then a must never appear.

For the full formula (disjunction of conjunctions):
- The formula is satisfied if **at least one** disjunction clause is satisfied
- A disjunction clause is satisfied if **all** its conjunction subformulas are satisfied

## Output 

The dataset is a list of tuples:
```python
[
    (['p1', 'p2', 'p3', 'p1', 'p2'], 1),  # Sequence satisfies formula
    (['p3', 'p2', 'p1', 'p3', 'p2'], 0),  # Sequence does not satisfy formula
    ...
]
```

**Note**: When saving to CSV, sequences are stored as comma-separated strings (e.g., `"p1,p2,p3,p1,p2"`).

## Example Output

Generated Formula:
((¬p1 U (p5 ∧ p3)) ∧ (¬p3 U (p2 ∧ p1))) ∨ ((¬p2 U (p1 ∧ p3)) ∧ (¬p5 U (p1 ∧ p3)))

Dataset Statistics:
  Total sequences: 100
  Positive (label 1): 50
  Negative (label 0): 50

Verifying correctness...
  Correct labels: 100/100 (100.00%)

Sample sequences:
  Positive examples:
    [p1,p3,p2,p2,p4,p1,p2,p2,p2,p2] - True
    [p4,p4,p1,p4,p3,p2,p3,p2,p5,p4] - True
  ...
```

