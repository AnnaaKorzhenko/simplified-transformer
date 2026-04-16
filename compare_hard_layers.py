#!/usr/bin/env python3
"""
Compare simplified-transformer (hard attention) with 1, 2, and 3 layers
on one dataset/split.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.utils import load_single_dataset, split_dataset_three_way
from training.train_transformer_hard import train_transformer_hard
from training.significance import mcnemar_chi2_paired


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="generated_diamond_star")
    p.add_argument("--formula_id", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="results_hard_layers.json")
    p.add_argument("--out_plot", default="plots_three_models/hard_layers_comparison.png")
    args = p.parse_args()

    _, dataset, alphabet, sequence_length = load_single_dataset(args.dataset_dir, args.formula_id)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=args.seed
    )

    results = {}
    for layers in (1, 2, 3):
        print(f"\n=== Training hard transformer with L={layers} ===")
        r = train_transformer_hard(
            X_train,
            y_train,
            X_test,
            y_test,
            alphabet,
            sequence_length,
            epochs=args.epochs,
            random_state=args.seed,
            num_layers=layers,
            X_val=X_val,
            y_val=y_val,
        )
        results[str(layers)] = {
            "accuracy": float(r["accuracy"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "auc": float(r["auc"]),
            "confusion_matrix": r["confusion_matrix"],
            "y_pred": r["y_pred"],
        }

    y_true = list(y_test)
    significance = {
        "method": "McNemar chi-square (Yates continuity), asymptotic p-value, same test split",
        "pairs": {},
    }
    for a, b in (("1", "2"), ("2", "3"), ("1", "3")):
        significance["pairs"][f"L{a}_vs_L{b}"] = mcnemar_chi2_paired(
            y_true, results[a]["y_pred"], results[b]["y_pred"]
        )

    payload = {
        "dataset_dir": args.dataset_dir,
        "formula_id": args.formula_id,
        "alphabet_size": len(alphabet),
        "sequence_length": sequence_length,
        "splits": "70/15/15 train/val/test, seed 42",
        "epochs": args.epochs,
        "readout": "begin-anchored (LLMsREs section 2): numpy trainer uses X[0,0]; torch SimplifiedTransformer uses X[:,0,0]",
        "results_by_layers": results,
        "significance_mcnemar": significance,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out_json}")

    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = range(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, layers in enumerate(("1", "2", "3")):
        vals = [results[layers][m] for m in metrics]
        shift = (idx - 1) * width
        ax.bar([i + shift for i in x], vals, width=width, label=f"L={layers}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Acc.", "Prec.", "Rec.", "F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Simplified Transformer (Hard) by Number of Layers")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out_plot}")


if __name__ == "__main__":
    main()

