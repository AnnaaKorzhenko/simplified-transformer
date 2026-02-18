# Simplified Transformer

## How to Run

```python
import torch
from transformer import SimplifiedTransformer

# Create model
model = SimplifiedTransformer(
    vocab_size=100,
    d_model=64,
    d_prime=32,
    num_layers=3,
    threshold=0.0
)

# Prepare input (must end with omega symbol)
batch_size = 4
seq_length = 10
vocab_size = 100

tokens = torch.randint(1, vocab_size - 1, (batch_size, seq_length - 1))
omega = torch.full((batch_size, 1), vocab_size - 1)
input_sequence = torch.cat([tokens, omega], dim=1)

# Run forward pass
output = model(input_sequence)  # Boolean predictions
```

### For Training

```python
raw_output = model.forward_raw(input_sequence)
loss = criterion(raw_output, labels.float())
```

### Requirements

- PyTorch >= 1.9.0
- Python 3.6+
