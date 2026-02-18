# Simplified Transformer

## How to Run

### Install Dependencies

```bash
pip install torch pandas
```

### Train

Train the transformer on a dataset:

```bash
python3 train.py --csv dataset_33.csv --json formula_33.json --epochs 50 --batch_size 32
```

Arguments:
- `--csv`: Path to CSV file with sequences
- `--json`: Path to JSON file with formula metadata
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--d_model`: Model dimension (default: 64)
- `--d_prime`: Query/key/value dimension (default: 32)
- `--num_layers`: Number of transformer layers (default: 3)
- `--lr`: Learning rate (default: 0.001)
- `--threshold`: Classification threshold (default: 0.0)

OR
Use the default training script:

```bash
python3 run_training.py
```

### Requirements

- PyTorch >= 1.9.0
- Python 3.6+
- pandas

