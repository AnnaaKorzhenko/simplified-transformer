"""
Generate large dataset for simplified transformer training:
- 50 formulas
- Alphabet size 20
- 50 positive sequences per formula
- 50 negative sequences per formula (for balanced dataset)
"""

from ltl_formulas.formula_generator import (
    FormulaGenerator,
    SyntheticDataGenerator,
    FormulaEvaluator,
    save_dataset_csv,
    save_formula_json
)
import json
import os

# Parameters
alphabet_size = 20
num_formulas = 50
num_positive = 50
num_negative = 50  
num_disjunctions = 2
num_conjunctions = 2
sequence_length = 10
seed = 42

# Create output directory (in root, one level up from training/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(root_dir, "large_datasets")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "formulas"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)

print("=" * 70)
print(f"Generating Large Dataset for Simplified Transformer")
print("=" * 70)
print(f"Parameters:")
print(f"  Alphabet size: {alphabet_size}")
print(f"  Number of formulas: {num_formulas}")
print(f"  Positive sequences per formula: {num_positive}")
print(f"  Negative sequences per formula: {num_negative}")
print(f"  Sequence length: {sequence_length}")
print(f"  Disjunctions per formula: {num_disjunctions}")
print(f"  Conjunctions per disjunction: {num_conjunctions}")
print("=" * 70)

all_results = []

for formula_id in range(1, num_formulas + 1):
    print(f"\n{'='*70}")
    print(f"Formula {formula_id}/{num_formulas}")
    print(f"{'='*70}")
    
    # Generate formula
    formula_gen = FormulaGenerator(
        alphabet_size=alphabet_size,
        num_disjunctions=num_disjunctions,
        num_conjunctions=num_conjunctions,
        seed=seed + formula_id
    )
    
    formula = formula_gen.generate_formula()
    print(f"Alphabet: {formula_gen.alphabet}")
    print(f"Formula: {formula_gen.formula_to_string(formula)}")
    
    # Generate dataset
    data_gen = SyntheticDataGenerator(
        alphabet=formula_gen.alphabet,
        seed=seed + formula_id
    )
    
    print(f"\nGenerating dataset...")
    dataset = data_gen.generate_dataset(
        formula=formula,
        sequence_length=sequence_length,
        num_positive=num_positive,
        num_negative=num_negative,
        seed=seed + formula_id
    )
    
    # Verify correctness
    positive_count = sum(1 for _, label in dataset if label == 1)
    negative_count = sum(1 for _, label in dataset if label == 0)
    
    correct = sum(
        1 for seq, label in dataset
        if FormulaEvaluator.evaluate_formula(seq, formula) == bool(label)
    )
    accuracy = correct / len(dataset) * 100
    
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Positive (label 1): {positive_count}")
    print(f"  Negative (label 0): {negative_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Save formula
    formula_filename = os.path.join(output_dir, "formulas", f"formula_{formula_id}.json")
    save_formula_json(formula, formula_filename, metadata={
        "formula_id": formula_id,
        "alphabet_size": alphabet_size,
        "num_disjunctions": num_disjunctions,
        "num_conjunctions": num_conjunctions,
        "alphabet": formula_gen.alphabet,
        "sequence_length": sequence_length,
        "num_positive": positive_count,
        "num_negative": negative_count
    })
    
    # Save dataset
    dataset_filename = os.path.join(output_dir, "datasets", f"dataset_{formula_id}.csv")
    save_dataset_csv(dataset, dataset_filename)
    
    print(f"\n  Saved formula to: {formula_filename}")
    print(f"  Saved dataset to: {dataset_filename}")
    
    # Store results summary
    all_results.append({
        "formula_id": formula_id,
        "formula": formula_gen.formula_to_string(formula),
        "alphabet": formula_gen.alphabet,
        "total_sequences": len(dataset),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accuracy": accuracy,
        "formula_file": formula_filename,
        "dataset_file": dataset_filename
    })

# Save summary
summary_filename = os.path.join(output_dir, "summary.json")
with open(summary_filename, 'w') as f:
    json.dump({
        "parameters": {
            "alphabet_size": alphabet_size,
            "num_formulas": num_formulas,
            "num_positive": num_positive,
            "num_negative": num_negative,
            "sequence_length": sequence_length,
            "num_disjunctions": num_disjunctions,
            "num_conjunctions": num_conjunctions
        },
        "results": all_results
    }, f, indent=2)

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"Generated {num_formulas} formulas with their datasets")
print(f"All files saved in directory: {output_dir}/")
print(f"Summary saved to: {summary_filename}")
print(f"\nFile structure:")
print(f"  {output_dir}/")
print(f"    formulas/formula_1.json ... formula_{num_formulas}.json")
print(f"    datasets/dataset_1.csv ... dataset_{num_formulas}.csv")
print(f"    summary.json")

# Display statistics
total_sequences = sum(r['total_sequences'] for r in all_results)
avg_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
print(f"\nOverall Statistics:")
print(f"  Total sequences: {total_sequences}")
print(f"  Average dataset accuracy: {avg_accuracy:.2f}%")

