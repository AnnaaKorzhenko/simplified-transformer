# Training Results on New Dataset (MTL-Compatible Semantics)

## Dataset Information

- **Source**: `generator_updated/dataset/`
- **Formula ID**: 33
- **Total Sequences**: 10,000
  - Positive: 5,000
  - Negative: 5,000
- **Alphabet Size**: 33
- **Sequence Length**: 5
- **Train/Test Split**: 80/20 (8,000 train, 2,000 test)

## Model Performance

### 1. Logistic Regression ✅

**Best Performance**

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.25%** |
| **Precision** | **98.53%** |
| **Recall** | **100.00%** |
| **F1 Score** | **99.26%** |
| **AUC-ROC** | **99.92%** |

**Confusion Matrix**:
```
[[979,  15],
 [  0, 1006]]
```

**Analysis**:
- Excellent performance across all metrics
- Perfect recall (100%) - correctly identifies all positive sequences
- Very high precision (98.53%) - few false positives
- Near-perfect AUC (99.92%) - excellent discrimination ability
- Only 15 false positives out of 2,000 test sequences

### 2. Transformer with Hard Attention ⚠️

**Poor Performance**

| Metric | Score |
|--------|-------|
| **Accuracy** | **49.90%** |
| **Precision** | **66.67%** |
| **Recall** | **0.80%** |
| **F1 Score** | **1.57%** |
| **AUC-ROC** | **37.60%** |

**Confusion Matrix**:
```
[[990,   4],
 [998,   8]]
```

**Analysis**:
- Performance barely better than random (50%)
- Very poor recall (0.80%) - misses almost all positive sequences
- High precision (66.67%) but only because it predicts very few positives
- Very low AUC (37.60%) - worse than random guessing
- Model is essentially predicting negative for almost everything

**Training Notes**:
- Trained for 100 epochs
- Train accuracy plateaued around 50%
- Evolutionary-style optimization may need more iterations or different parameters

### 3. Transformer with Softmax Attention ⚠️

**Poor Performance**

| Metric | Score |
|--------|-------|
| **Accuracy** | **33.85%** |
| **Precision** | **27.39%** |
| **Recall** | **19.09%** |
| **F1 Score** | **22.50%** |
| **AUC-ROC** | **32.69%** |

**Confusion Matrix**:
```
[[485, 509],
 [814, 192]]
```

**Analysis**:
- Performance worse than random (50%)
- Poor recall (19.09%) - misses most positive sequences
- Low precision (27.39%) - many false positives
- Very low AUC (32.69%) - worse than random guessing
- Model struggles significantly with this task

**Training Notes**:
- Trained for 100 epochs
- Train accuracy plateaued around 34%
- Loss remained high (~1.036)
- Gradient descent may need different learning rate or architecture

## Comparison Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Logistic Regression** | **99.25%** | **98.53%** | **100.00%** | **99.26%** | **99.92%** |
| Transformer (Hard) | 49.90% | 66.67% | 0.80% | 1.57% | 37.60% |
| Transformer (Softmax) | 33.85% | 27.39% | 19.09% | 22.50% | 32.69% |

## Key Observations

1. **Logistic Regression excels**: The feature engineering (positional one-hot encoding, frequency counts, first/last occurrence positions) captures the temporal logic patterns effectively.

2. **Transformers struggle**: Both transformer variants perform poorly, suggesting:
   - The simplified architecture may not be sufficient for this task
   - Training may need more epochs or better hyperparameters
   - The attention mechanism may not be learning the relevant patterns
   - The gradient-free optimization (hard attention) and gradient descent (softmax) may both need tuning

3. **MTL-compatible semantics**: The dataset with corrected semantics (100% MTL verification accuracy) shows that logistic regression can learn the patterns effectively, while transformers need more work.

## Recommendations

1. **For production use**: Use Logistic Regression - it achieves excellent performance (99.25% accuracy).

2. **For transformer improvement**:
   - Increase training epochs
   - Tune hyperparameters (learning rate, architecture dimensions)
   - Try different attention mechanisms
   - Consider pre-training or different initialization strategies
   - Experiment with different loss functions

3. **Dataset**: The new dataset with MTL-compatible semantics is ready for further experimentation.

## Files

- Results: `generator_updated/results/all_models_results.json`
- ROC curves: Saved in `generator_updated/results/`
- Individual model results: Saved in `generator_updated/results/`

