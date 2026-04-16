#!/usr/bin/env python3
"""
Train simplified hard-attention transformer on two semantics settings and compare.

1) **seen_pair**: (¬a) U ((⊤ U b) ∧ (⊤ U c)) — `generated_diamond_star`.
2) **until_single**: (¬a) U (⊤ U b) with witness b — `generated_until_single`.

Same split protocol, epochs, and layer grid (1–3) for both; writes one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os

from training.utils import load_single_dataset, split_dataset_three_way
from training.train_transformer_hard import train_transformer_hard
from training.significance import mcnemar_chi2_paired


def _significance_for_layers(by_layers: dict, y_true: list) -> dict:
    out = {
        "method": "McNemar (same test set), Yates continuity, asymptotic p-value",
        "pairs": {},
    }
    keys = sorted(by_layers.keys(), key=lambda k: int(k))
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            out["pairs"][f"L{a}_vs_L{b}"] = mcnemar_chi2_paired(
                y_true, by_layers[a]["y_pred"], by_layers[b]["y_pred"]
            )
    return out


def run_setting(
    dataset_dir: str,
    formula_id: int,
    epochs: int,
    seed: int,
    layers_list: tuple[int, ...],
) -> dict:
    _, dataset, alphabet, sequence_length = load_single_dataset(dataset_dir, formula_id)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=seed
    )
    by_layers = {}
    for layers in layers_list:
        print(f"\n=== {dataset_dir} L={layers} ===")
        r = train_transformer_hard(
            X_train,
            y_train,
            X_test,
            y_test,
            alphabet,
            sequence_length,
            epochs=epochs,
            random_state=seed,
            num_layers=layers,
            X_val=X_val,
            y_val=y_val,
        )
        by_layers[str(layers)] = {
            "accuracy": float(r["accuracy"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "auc": float(r["auc"]),
            "confusion_matrix": r["confusion_matrix"],
            "y_pred": r["y_pred"],
        }
    y_list = list(y_test)
    sig = _significance_for_layers(by_layers, y_list)
    return {
        "dataset_dir": dataset_dir,
        "formula_id": formula_id,
        "alphabet_size": len(alphabet),
        "sequence_length": sequence_length,
        "y_test": y_list,
        "significance_mcnemar": sig,
        "results_by_layers": by_layers,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seen_pair_dir",
        default="generated_diamond_star",
        help="Dataset dir for (¬a U ((⊤ U b) ∧ (⊤ U c))) / SeenPair in tooling",
    )
    p.add_argument(
        "--until_single_dir",
        default="generated_until_single",
        help="Dataset dir for (¬a U (⊤ U b))",
    )
    p.add_argument("--formula_id", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default="results_semantics_seen_pair_vs_until_single.json",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="1,2,3",
        help="Comma-separated layer counts, e.g. 1,2,3",
    )
    args = p.parse_args()

    layers_tuple = tuple(int(x.strip()) for x in args.layers.split(",") if x.strip())
    seen = run_setting(
        args.seen_pair_dir, args.formula_id, args.epochs, args.seed, layers_tuple
    )
    single = run_setting(
        args.until_single_dir, args.formula_id, args.epochs, args.seed, layers_tuple
    )

    payload = {
        "protocol": "70/15/15 train/val/test, seed matches --seed",
        "epochs": args.epochs,
        "layers": list(layers_tuple),
        "formula_id": args.formula_id,
        "seen_pair": {
            "description": "(¬a) U ((⊤ U b) ∧ (⊤ U c)); BLACK uses derived seenpair_b_c",
            **seen,
        },
        "until_single": {
            "description": "(¬a) U (⊤ U b) with atomic witness b",
            **single,
        },
        "note_cross_setting": "McNemar applies only within each row (same y_test). "
        "Comparing seen_pair vs until_single accuracies is not a paired test on identical labels.",
    }
    out_abs = os.path.abspath(args.out_json)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
