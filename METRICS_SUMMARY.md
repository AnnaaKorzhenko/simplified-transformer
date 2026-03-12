# Training metrics summary (simplified transformer, leftmost attention)

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|-----|---------|
| **Diamond-star (♢⋆) synthetic** (`generated_diamond_star/datasets/dataset_1`) | 0.7400 | 0.7738 | 0.7647 | 0.7692 | 0.7917 |
| **UCI Poker Hand binary** (pair-or-better vs high-card, 200k) | 0.6761 | 0.6266 | 0.8654 | 0.7269 | 0.7965 |
| **Balanced parentheses** (leftmost-marked features, 2k) | 0.5733 | 0.5299 | 0.8732 | 0.6596 | 0.6980 |

- **Diamond-star**: LTL formula task with ♢⋆(φ) (formula holds at start); labels verified with external MTL checker.
- **UCI Poker**: Binary = 0 (high card only) / 1 (pair or better). Data: [UCI Poker Hand](https://archive.ics.uci.edu/ml/datasets/poker%2Bhand).
- **Balanced parentheses**: Binary label from balanced/unbalanced; sequences adapted with `LM|len=…|sum=…|min=…|max=…` for leftmost attention.

Datasets are stored in **parquet** under `generated_diamond_star/datasets/`, `datasets_poker_uci/`, and root (`balanced_parens_adapted_lm.parquet`). Full result JSONs: `results_diamond_star.json`, `results_uci_poker_binary.json`.
