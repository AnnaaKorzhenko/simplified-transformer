import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formula + dataset with ♢⋆ wrapper in metadata")
    parser.add_argument("--output_dir", type=str, default="generated_diamond_star", help="Output directory")
    parser.add_argument("--formula_id", type=int, default=1, help="ID used in saved filenames/metadata")
    parser.add_argument("--alphabet_size", type=int, default=10)
    parser.add_argument("--num_disjunctions", type=int, default=2)
    parser.add_argument("--num_conjunctions", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--num_positive", type=int, default=500)
    parser.add_argument("--num_negative", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reuse the existing generator implementation (keeps semantics consistent with your repo).
    upstream_dir = os.path.join(os.path.dirname(__file__), "simplified-transformer-upstream", "ltl_formulas")
    sys.path.insert(0, upstream_dir)
    from formula_generator import (  # noqa: E402
        FormulaGenerator,
        SyntheticDataGenerator,
        FormulaEvaluator,
        save_formula_json,
        save_dataset_csv,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    formulas_dir = os.path.join(args.output_dir, "formulas")
    datasets_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(formulas_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    formula_gen = FormulaGenerator(
        alphabet_size=args.alphabet_size,
        num_disjunctions=args.num_disjunctions,
        num_conjunctions=args.num_conjunctions,
        seed=args.seed,
    )
    formula = formula_gen.generate_formula()
    inner = formula_gen.formula_to_string(formula)

    # Diamond-star is a begin-of-sequence anchoring operator (LLMsREs_12).
    # Under this repo's convention (evaluate satisfaction at position 1), ♢⋆(φ) is equivalent to φ.
    formula_str_diamond_star = f"♢⋆({inner})"
    formula_str_mtl_compatible = inner  # best-effort for external checkers that don't support ♢⋆

    data_gen = SyntheticDataGenerator(alphabet=formula_gen.alphabet, seed=args.seed)
    dataset = data_gen.generate_dataset(
        formula=formula,
        sequence_length=args.sequence_length,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed,
    )

    # Verify labels (should be ~100% for the generator's own semantics).
    correct = 0
    for seq, label in dataset:
        if FormulaEvaluator.evaluate_formula(seq, formula) == bool(label):
            correct += 1
    acc = correct / len(dataset) if dataset else 0.0

    formula_path = os.path.join(formulas_dir, f"formula_{args.formula_id}.json")
    dataset_path = os.path.join(datasets_dir, f"dataset_{args.formula_id}.csv")

    save_formula_json(
        formula,
        formula_path,
        metadata={
            "formula_id": args.formula_id,
            "alphabet_size": args.alphabet_size,
            "num_disjunctions": args.num_disjunctions,
            "num_conjunctions": args.num_conjunctions,
            "alphabet": formula_gen.alphabet,
            "sequence_length": args.sequence_length,
            "num_positive": args.num_positive,
            "num_negative": args.num_negative,
            "total_sequences": len(dataset),
            "seed": args.seed,
            "formula_string": inner,
            "formula_string_diamond_star": formula_str_diamond_star,
            "formula_string_mtl_compatible": formula_str_mtl_compatible,
            "diamond_star": True,
            "label_verification_accuracy": acc,
        },
    )
    save_dataset_csv(dataset, dataset_path)

    print("Saved:")
    print(" ", formula_path)
    print(" ", dataset_path)
    print(f"Label verification accuracy: {acc:.4f}")
    print("\nFormula (diamond-star):")
    print(formula_str_diamond_star)


if __name__ == "__main__":
    main()

