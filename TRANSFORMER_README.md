# Simplified Transformer

## How to Run

### Install Dependencies

```bash
pip install torch pandas scikit-learn numpy matplotlib seaborn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Run Example

Run the example script to see the transformer in action:

```bash
python3 example.py
```

### Train the Model

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

### Quick Training

Use the default training script:

```bash
python3 run_training.py
```

### Requirements

- PyTorch >= 1.9.0
- Python 3.6+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
