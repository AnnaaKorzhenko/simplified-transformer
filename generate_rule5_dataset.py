#!/usr/bin/env python3
"""Generate formula JSON + CSV dataset for LLMsREs rule (5) style (see ltl_formulas/rule5_formula.py)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltl_formulas.formula_generator import save_dataset_csv  # noqa: E402
from ltl_formulas.rule5_formula import (  # noqa: E402
    Rule5SyntheticGenerator,
    save_rule5_formula_json,
    evaluate_rule5_instance,
    rule5_to_paper_string,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="generated_rule5_llmsres")
    p.add_argument("--formula_id", type=int, default=1)
    p.add_argument("--alphabet_size", type=int, default=6)
    p.add_argument("--sequence_length", type=int, default=10)
    p.add_argument("--num_positive", type=int, default=500)
    p.add_argument("--num_negative", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    alphabet = [f"p{i+1}" for i in range(args.alphabet_size)]
    rule = {
        "head": "p1",
        "witness": "p2",
        "beta_same_partition": ["p3", "p4"],
        "never_symbols": ["p5"],
    }
    for key in ("head", "witness"):
        if rule[key] not in alphabet:
            raise ValueError("Rule symbols must lie in alphabet")
    for s in rule["beta_same_partition"] + rule["never_symbols"]:
        if s not in alphabet:
            raise ValueError(f"Symbol {s} not in alphabet")

    os.makedirs(args.output_dir, exist_ok=True)
    formulas_dir = os.path.join(args.output_dir, "formulas")
    datasets_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(formulas_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    formula_path = os.path.join(formulas_dir, f"formula_{args.formula_id}.json")
    dataset_path = os.path.join(datasets_dir, f"dataset_{args.formula_id}.csv")

    save_rule5_formula_json(
        formula_path,
        rule,
        alphabet,
        args.sequence_length,
        description="Synthetic rule (5): head at pos 0, ♢⋆ conjuncts at trace start (see rule5_formula.py).",
    )

    gen = Rule5SyntheticGenerator(alphabet, rule, seed=args.seed)
    dataset = gen.generate_dataset(
        args.sequence_length, args.num_positive, args.num_negative
    )
    save_dataset_csv(dataset, dataset_path)

    bad = sum(
        1 for seq, lab in dataset if evaluate_rule5_instance(seq, rule) != bool(lab)
    )
    print(rule5_to_paper_string(rule))
    print(f"Wrote {formula_path}")
    print(f"Wrote {dataset_path}")
    print(f"Label check errors: {bad} / {len(dataset)}")


if __name__ == "__main__":
    main()
