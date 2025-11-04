# Training Results with Train/Validation/Test Split

## Dataset Split

- **Train**: 7,000 sequences (70%)
- **Validation**: 1,500 sequences (15%)
- **Test**: 1,500 sequences (15%)

## Model Performance

### 1. Logistic Regression ✅

**Excellent Performance**

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **99.53%** |
| **Test Accuracy** | **99.13%** |
| **Precision** | **98.31%** |
| **Recall** | **100.00%** |
| **F1 Score** | **99.15%** |
| **AUC-ROC** | **99.91%** |

**Confusion Matrix**:
```
[[732,  13],
 [  0, 755]]
```

**Analysis**:
- Excellent performance on both validation and test sets
- Perfect recall (100%) - no false negatives
- Very high precision (98.31%) - only 13 false positives
- Consistent performance between validation and test sets

### 2. Transformer with Hard Attention ⚠️

**Poor Performance**

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **51.73%** (best during training) |
| **Test Accuracy** | **49.67%** |
| **Precision** | **0.00%** |
| **Recall** | **0.00%** |
| **F1 Score** | **0.00%** |
| **AUC-ROC** | **34.96%** |

**Confusion Matrix**:
```
[[745,   0],
 [755,   0]]
```

**Analysis**:
- Model predicts all sequences as negative (class 0)
- Performance barely above random (50%)
- Validation accuracy reached 51.73% but test performance is worse
- Model fails to learn meaningful patterns

**Training Notes**:
- Best validation accuracy: 51.73% at epochs 40-80
- Model selection based on validation accuracy
- Still performs poorly on test set

### 3. Transformer with Softmax Attention ⚠️

**Poor Performance**

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **34.53%** (best during training) |
| **Test Accuracy** | **33.07%** |
| **Precision** | **26.38%** |
| **Recall** | **18.41%** |
| **F1 Score** | **21.68%** |
| **AUC-ROC** | **31.47%** |

**Confusion Matrix**:
```
[[357, 388],
 [616, 139]]
```

**Analysis**:
- Performance worse than random (50%)
- Low recall (18.41%) - misses most positive sequences
- Low precision (26.38%) - many false positives
- Very low AUC (31.47%) - worse than random guessing

**Training Notes**:
- Best validation accuracy: 34.53% at epoch 80
- Early stopping implemented but not triggered (no improvement threshold)
- Loss remained high (~1.036) throughout training
- Model shows minimal learning

## Comparison Summary

| Model | Val Accuracy | Test Accuracy | Precision | Recall | F1 | AUC |
|-------|-------------|---------------|-----------|--------|-----|-----|
| **Logistic Regression** | **99.53%** | **99.13%** | **98.31%** | **100.00%** | **99.15%** | **99.91%** |
| Transformer (Hard) | 51.73% | 49.67% | 0.00% | 0.00% | 0.00% | 34.96% |
| Transformer (Softmax) | 34.53% | 33.07% | 26.38% | 18.41% | 21.68% | 31.47% |

## Key Observations

1. **Logistic Regression excels**: Maintains excellent performance with train/val/test split
   - Validation accuracy: 99.53%
   - Test accuracy: 99.13%
   - Consistent performance across splits

2. **Transformers struggle**: Both variants perform poorly
   - Hard Attention: Validation shows slight improvement (51.73%) but test performance is worse
   - Softmax: Validation shows minimal improvement (34.53%) but test performance is worse
   - Both models show overfitting or inability to generalize

3. **Validation set usefulness**: 
   - Helps identify best model checkpoints
   - Shows that transformers are not learning effectively
   - Logistic Regression shows consistent performance across all splits

## Recommendations

1. **For production use**: Use Logistic Regression - maintains excellent performance with proper validation

2. **For transformer improvement**:
   - Need better architecture or initialization
   - Current training methods (evolutionary for hard, approximate gradients for softmax) may be insufficient
   - Consider more sophisticated optimization techniques
   - May need different loss functions or training strategies

3. **Dataset**: The train/val/test split provides better model evaluation and helps identify overfitting issues.

## Files

- Results: `generator_updated/results/all_models_results.json`
- Training log: `generator_updated/training_output_3way_split.log`
- ROC curves: Saved in `generator_updated/results/`

