# Updated Generator with MTL-Compatible Semantics

This folder contains the updated formula generator and dataset that achieves **100% accuracy** with the [py-metric-temporal-logic](https://github.com/mvcisback/py-metric-temporal-logic) library.

## What's Included

### Core Files
- **`ltl_formulas/formula_generator.py`**: Updated evaluator with MTL-compatible semantics
- **`dataset/formula_33.json`**: Formula 33 with metadata
- **`dataset/dataset_33.csv`**: 10,000 sequences (5,000 positive, 5,000 negative) with corrected labels

### Scripts
- **`generate_large_single_dataset.py`**: Script to generate the large dataset
- **`verify_with_mtl.py`**: Verification script using py-metric-temporal-logic library

## Key Updates

### Semantic Changes

1. **"Simultaneously True" Semantics**: Updated to match MTL semantics where `(b & c)` means both symbols have appeared, using the later position as the "simultaneous" point.

2. **Edge Case Fix**: Fixed the case where `b` or `c` is missing:
   - **Old**: Returned `True` if `a` was missing (incorrect)
   - **New**: Returns `False` because `(b & c)` is never True (matches MTL)

3. **"Seen So Far" Semantics**: MTL verification uses "seen so far" interpretation where symbols become True at first appearance and remain True.

## Dataset Details

- **Formula ID**: 33
- **Alphabet Size**: 33
- **Disjunctions**: 2
- **Conjunctions per Disjunction**: 2
- **Sequence Length**: 5
- **Total Sequences**: 10,000
  - Positive: 5,000
  - Negative: 5,000
- **Seed**: 42 (for reproducibility)

## Verification Results

✅ **100% Accuracy** when verified with py-metric-temporal-logic library

- Total sequences checked: 10,000
- Correct labels: 10,000
- Incorrect labels: 0

## Usage

### Generate Dataset
```bash
python3 generate_large_single_dataset.py
```

### Verify with MTL
```bash
python3 verify_with_mtl.py
```

### Use the Generator
```python
from ltl_formulas.formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator
)

# Generate formula
formula_gen = FormulaGenerator(
    alphabet_size=33,
    num_disjunctions=2,
    num_conjunctions=2,
    seed=42
)
formula = formula_gen.generate_formula()

# Generate data
data_gen = SyntheticDataGenerator(
    alphabet=formula_gen.alphabet,
    seed=42
)
dataset = data_gen.generate_dataset(
    formula=formula,
    sequence_length=5,
    num_positive=5000,
    num_negative=5000,
    seed=42
)

# Evaluate
result = FormulaEvaluator.evaluate_formula(sequence, formula)
```

## Technical Details

### MTL Compatibility

The evaluator now perfectly matches MTL semantics:
- `(~a U (b & c))` evaluates correctly when `b` or `c` is missing
- Uses "seen so far" semantics for MTL verification
- Handles edge cases identically to MTL library

### Sequence Model

Our sequence model uses **one symbol per position**. To match MTL's "simultaneously true" semantics:
- Symbols become True at their first appearance
- They remain True thereafter ("seen so far")
- This allows `(b & c)` to be True when both have appeared

## Files Structure

```
generator_updated/
├── README.md
├── generate_large_single_dataset.py
├── verify_with_mtl.py
├── ltl_formulas/
│   ├── __init__.py
│   ├── formula_generator.py  # Updated with MTL-compatible semantics
│   └── ...
└── dataset/
    ├── formula_33.json
    └── dataset_33.csv
```

