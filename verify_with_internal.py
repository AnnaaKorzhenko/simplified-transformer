"""
Verify sequences against LTL formulas using the project's internal FormulaEvaluator.

This is the canonical LTL checker: same semantics as dataset generation (see doc_synthetic_dataset.tex).
Supports the formula class: disjunction of conjunctions of (¬a U (b ∧ c)).
No external dependency (no MTL); consistent with past-operator–ready semantics.
"""

import os
import json
import csv
from typing import List, Tuple

from ltl_formulas.formula_generator import load_formula_json, sequence_satisfies_payload


def verify_sequence(sequence: List[str], formula: list, metadata: dict) -> bool:
    """Return True iff the sequence satisfies the formula (payload may be rule5 or until-atoms)."""
    return sequence_satisfies_payload(sequence, formula, metadata)


def verify_full_dataset(formula_file: str, dataset_file: str) -> dict:
    """
    Verify all sequences in the dataset using the internal FormulaEvaluator.
    """
    formula, metadata = load_formula_json(formula_file)
    alphabet = metadata.get("alphabet", [])

    if not os.path.exists(dataset_file):
        return {"error": f"Dataset file not found: {dataset_file}"}

    dataset = []
    with open(dataset_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = [s.strip() for s in row["sequence"].split(",")]
            label = int(row["label"])
            dataset.append((sequence, label))

    correct = 0
    errors = []
    for i, (seq, expected_label) in enumerate(dataset):
        actual = sequence_satisfies_payload(seq, formula, metadata)
        expected = bool(expected_label)
        if actual == expected:
            correct += 1
        else:
            if len(errors) < 10:
                errors.append({
                    "sequence": ",".join(seq),
                    "expected": expected_label,
                    "actual": 1 if actual else 0,
                })

    total = len(dataset)
    accuracy = (correct / total * 100.0) if total else 0.0
    kind = metadata.get("formula_kind", "pair_witness_until")
    method = (
        "sequence_satisfies_payload: rule5_llmsres"
        if kind == "rule5_llmsres"
        else "sequence_satisfies_payload: pair-witness until (FormulaEvaluator)"
    )

    return {
        "total_checked": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "errors": errors[:10],
        "verification_method": method,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify dataset with internal LTL checker")
    parser.add_argument("--formula", "-f", default="generated_diamond_star/formulas/formula_1.json", help="Formula JSON path")
    parser.add_argument("--dataset", "-d", default="generated_diamond_star/datasets/dataset_1.csv", help="Dataset CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("Internal LTL checker (FormulaEvaluator)")
    print("=" * 60)
    print(f"Formula: {args.formula}")
    print(f"Dataset: {args.dataset}")

    if not os.path.exists(args.formula):
        print(f"Error: Formula file not found: {args.formula}")
        return

    formula, meta = load_formula_json(args.formula)
    if meta.get("formula_kind") == "rule5_llmsres":
        print(f"\nFormula (rule5): {meta.get('rule5_string', '')}")
    else:
        from ltl_formulas.formula_generator import FormulaGenerator
        gen = FormulaGenerator(1, 1, 1)
        gen.alphabet = meta.get("alphabet", [])
        print(f"\nFormula: {gen.formula_to_string(formula)}")

    results = verify_full_dataset(args.formula, args.dataset)
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("Verification results")
    print("=" * 60)
    print(f"Method: {results['verification_method']}")
    print(f"Total: {results['total_checked']}  Correct: {results['correct']}  Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    if results["errors"]:
        print("\nFirst errors:")
        for e in results["errors"]:
            print(f"  expected {e['expected']}, got {e['actual']}: {e['sequence'][:60]}...")

    out_file = "verification_internal_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
