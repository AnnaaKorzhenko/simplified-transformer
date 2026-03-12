# LTL Formula Extractor

## Overview

The `extractor.py` module implements the LTL formula extraction algorithm from the PDF "Extracting LTL formulas from Transformers". It converts a trained Simplified Transformer into an equivalent LTL (Linear Temporal Logic) formula.

## Algorithm

Based on the PDF, the extraction process:

1. **Rule Extraction**: For each layer ℓ, extract rules of the form:
   ```
   a ∧ φ → a'
   ```
   where:
   - `a` is a symbol from layer ℓ-1
   - `φ` is an LTL condition involving operators like `(¬b U a)` and `(⊤ U b)`
   - `a'` is the resulting symbol in layer ℓ

2. **Attention Score Computation**: For each pair of symbols (a₁, a₂), compute:
   ```
   α_{a₁,a₂} = x_{ℓ-1,a₁} Q^ℓ (K^ℓ)^T x_{ℓ-1,a₂}^T
   ```

3. **Symbol Partitioning**: Partition symbols based on attention scores, ordered by decreasing score.

4. **Rule Construction**: Create rules where (LLMsREs_12 update):
   - Conditions are wrapped in **diamond-star** `♢⋆(·)` to indicate they are evaluated at the **beginning of the sequence** (global sentence semantics).
   - Symbols in higher partitions (higher attention) appear as `¬(⊤ U b')`
   - Symbols in the same partition appear as `(¬b' U b)` (or `(⊤ U b)` if the partition is a singleton)

5. **Unfolding**: Recursively unfold rules using the `(·)°` operator to get the final formula.

6. **Final Formula**: The formula Φ_T is a disjunction of unfolded rules where the head symbol leads to acceptance (value > threshold).

## Usage

```python
from transformer import SimplifiedTransformer
from extractor import LTLExtractor
import json

# Load trained model
model = SimplifiedTransformer(...)
model.load_state_dict(torch.load('best_model.pt'))

# Load alphabet
with open('formula_33.json', 'r') as f:
    formula_data = json.load(f)
alphabet = formula_data['metadata']['alphabet']

# Create extractor
extractor = LTLExtractor(model, alphabet, threshold=0.0)

# Extract formula
result = extractor.extract_formula()
print(f"Extracted formula: {result['formula']}")
```

## Testing

Run the test script:

```bash
python3 test_extractor.py
```

This will:
1. Load the trained model (if available)
2. Extract the LTL formula
3. Show extracted rules by layer
4. Test model predictions on sample sequences

## Implementation Details

### Key Classes

- **LTLExtractor**: Main extractor class that implements the extraction algorithm

### Key Methods

- `compute_attention_scores()`: Computes attention scores between symbol pairs
- `partition_symbols_by_attention()`: Partitions symbols based on attention scores
- `create_rule()`: Creates a rule with LTL conditions
- `extract_layer_rules()`: Extracts all rules for a given layer
- `extract_formula()`: Main extraction method that returns the final formula

### LTL Operators

The extractor uses standard LTL operators:
- `(¬b U a)`: "a occurs before b" (a until b doesn't occur)
- `(⊤ U b)`: "b eventually occurs"
- `¬(⊤ U b)`: "b never occurs"

## Limitations

The current implementation is a simplified version. Full extraction requires:

1. **Complete Vector Enumeration**: Enumerating all possible vectors that can appear in each layer (computationally expensive)

2. **Full Unfolding**: Complete recursive unfolding of all rules through all layers

3. **Symbol Tracking**: Proper tracking of how symbols transform through layers

For production use, consider:
- Sampling-based approaches for vector enumeration
- Caching of computed attention scores
- Optimization for large vocabularies

## References

Based on: "Extracting LTL formulas from Transformers" (LLMsREs_12.pdf). The paper defines:

- `♢⋆(φ)` as a macro for “\(φ\) holds at the first time point”.
