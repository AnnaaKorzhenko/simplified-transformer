# Dataset Verification Report

## Dataset Information

- **Location**: `generator_updated/dataset/`
- **Formula ID**: 33
- **Total Sequences**: 10,000
- **Positive Sequences**: 5,000
- **Negative Sequences**: 5,000
- **Alphabet Size**: 33
- **Sequence Length**: 5

## Verification Results

### Internal Evaluator Verification ✅

**100.00% Accuracy**

- All 10,000 sequences verified with our `FormulaEvaluator`
- 0 incorrect labels
- All sequences generated match their labels perfectly

### MTL Library Verification ✅

**100.00% Accuracy** (with "seen so far" semantics)

- Verified using [py-metric-temporal-logic](https://github.com/mvcisback/py-metric-temporal-logic) library
- All sequences match MTL semantics when using "seen so far" interpretation
- Perfect alignment between our evaluator and MTL library

## Training Results

All models trained with **70/15/15 train/validation/test split**:

### 1. Logistic Regression ✅

| Metric | Score |
|--------|-------|
| Validation Accuracy | **99.53%** |
| Test Accuracy | **99.13%** |
| Precision | **98.31%** |
| Recall | **100.00%** |
| F1 Score | **99.15%** |
| AUC-ROC | **99.91%** |

### 2. Transformer (Hard Attention) ⚠️

| Metric | Score |
|--------|-------|
| Validation Accuracy | 51.73% |
| Test Accuracy | 49.67% |
| Precision | 0.00% |
| Recall | 0.00% |
| F1 Score | 0.00% |
| AUC-ROC | 34.96% |

### 3. Transformer (Softmax Attention) ⚠️

| Metric | Score |
|--------|-------|
| Validation Accuracy | 34.53% |
| Test Accuracy | 33.07% |
| Precision | 26.38% |
| Recall | 18.41% |
| F1 Score | 21.68% |
| AUC-ROC | 31.47% |

## Dataset Quality

- ✅ **100% label accuracy** - All sequences correctly labeled
- ✅ **MTL-compatible** - Matches MTL semantics exactly
- ✅ **Balanced** - Equal number of positive and negative examples
- ✅ **Verified** - Double-checked with both internal and external evaluators

## Files

- **Formula**: `generator_updated/dataset/formula_33.json`
- **Dataset**: `generator_updated/dataset/dataset_33.csv`
- **Training Results**: `generator_updated/results/`

