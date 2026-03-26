#!/usr/bin/env python3
"""
Train simplified transformer (hard leftmost attention), softmax-attention variant,
and logistic regression on the same splits for: diamond-star synthetic, UCI poker (subsample),
and balanced parentheses.

Writes results_three_models.json and plots under plots_three_models/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from training.utils import split_dataset_three_way
from training.train_transformer_hard import train_transformer_hard
from training.train_transformer_softmax import train_transformer_soft
from training.train_logistic_regression import train_logistic_regression
from training.utils import load_single_dataset


def load_csv_tokens(
    csv_path: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[List[Tuple[List[str], int]], List[str], int]:
    """Load dataset with columns sequence,label. Sequence is comma-separated or single string."""
    rows: List[Tuple[List[str], int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = row["sequence"].strip()
            if "," in s:
                seq = [t.strip() for t in s.split(",") if t.strip()]
            else:
                seq = list(s)
            rows.append((seq, int(row["label"])))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]
    tok_set = set()
    max_len = 0
    for seq, _ in rows:
        max_len = max(max_len, len(seq))
        tok_set.update(seq)
    alphabet = sorted(tok_set)
    return rows, alphabet, max_len


def run_one_dataset(
    name: str,
    dataset: List[Tuple[List[str], int]],
    alphabet: List[str],
    sequence_length: int,
    epochs_hard: int,
    epochs_soft: int,
    seed: int = 42,
) -> dict:
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=seed
    )
    out = {
        "dataset": name,
        "n_total": len(dataset),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "alphabet_size": len(alphabet),
        "sequence_length": sequence_length,
        "models": {},
    }
    print(f"\n{'='*60}\n{name}: hard transformer\n{'='*60}")
    r_hard = train_transformer_hard(
        X_train,
        y_train,
        X_test,
        y_test,
        alphabet,
        sequence_length,
        epochs=epochs_hard,
        random_state=seed,
        X_val=X_val,
        y_val=y_val,
    )
    out["models"]["hard"] = {
        "accuracy": float(r_hard["accuracy"]),
        "precision": float(r_hard["precision"]),
        "recall": float(r_hard["recall"]),
        "f1": float(r_hard["f1"]),
        "auc": float(r_hard["auc"]),
        "confusion_matrix": r_hard["confusion_matrix"],
    }

    print(f"\n{'='*60}\n{name}: softmax transformer\n{'='*60}")
    r_soft = train_transformer_soft(
        X_train,
        y_train,
        X_test,
        y_test,
        alphabet,
        sequence_length,
        epochs=epochs_soft,
        learning_rate=0.001,
        random_state=seed,
        X_val=X_val,
        y_val=y_val,
    )
    out["models"]["softmax"] = {
        "accuracy": float(r_soft["accuracy"]),
        "precision": float(r_soft["precision"]),
        "recall": float(r_soft["recall"]),
        "f1": float(r_soft["f1"]),
        "auc": float(r_soft["auc"]),
        "confusion_matrix": r_soft["confusion_matrix"],
    }

    print(f"\n{'='*60}\n{name}: logistic regression\n{'='*60}")
    r_lr = train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        alphabet,
        sequence_length,
        X_val=X_val,
        y_val=y_val,
        random_state=seed,
    )
    out["models"]["logistic_regression"] = {
        "accuracy": float(r_lr["accuracy"]),
        "precision": float(r_lr["precision"]),
        "recall": float(r_lr["recall"]),
        "f1": float(r_lr["f1"]),
        "auc": float(r_lr["auc"]),
        "confusion_matrix": r_lr["confusion_matrix"],
    }
    return out, r_hard, r_soft, r_lr, y_test


def plot_metrics_bars(block: dict, out_path: str) -> None:
    models = ["hard", "softmax", "logistic_regression"]
    labels = ["Hard-attn\n(simplified)", "Softmax\n(transformer)", "Logistic\nregression"]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        vals = [block["models"][m][k] for k in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Acc.", "Prec.", "Rec.", "F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(block["dataset"])
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_three(y_test, r_hard, r_soft, r_lr, title: str, out_path: str) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    curves = [
        ("Hard-attn (simplified)", r_hard["y_pred_proba"], r_hard["auc"], "green"),
        ("Softmax transformer", r_soft["y_pred_proba"], r_soft["auc"], "red"),
        ("Logistic regression", r_lr["y_pred_proba"], r_lr["auc"], "blue"),
    ]
    for name, probs, auc, color in curves:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={float(auc):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_diamond_hard", type=int, default=100)
    parser.add_argument("--epochs_diamond_soft", type=int, default=100)
    parser.add_argument("--epochs_other_hard", type=int, default=80)
    parser.add_argument("--epochs_other_soft", type=int, default=50)
    parser.add_argument("--uci_max", type=int, default=5000, help="Max rows for UCI (speed)")
    parser.add_argument("--uci_csv", type=str, default="datasets_poker_uci/uci_pokerhand_binary_200k.csv")
    parser.add_argument("--parens_csv", type=str, default="balanced_parens.csv")
    parser.add_argument("--out_dir", type=str, default="plots_three_models")
    parser.add_argument("--json_out", type=str, default="results_three_models.json")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    triples = []

    # 1) Diamond-star
    _f, dataset, alphabet, seq_len = load_single_dataset("generated_diamond_star", 1)
    block, rh, rs, rl, yt = run_one_dataset(
        "diamond_star_synthetic",
        dataset,
        alphabet,
        seq_len,
        args.epochs_diamond_hard,
        args.epochs_diamond_soft,
    )
    triples.append((block, rh, rs, rl, yt))

    # 2) UCI poker (subsample)
    uci_path = args.uci_csv if os.path.isabs(args.uci_csv) else os.path.join(root, args.uci_csv)
    ds_uci, alpha_uci, sl_uci = load_csv_tokens(uci_path, max_samples=args.uci_max)
    block_u, rh_u, rs_u, rl_u, yt_u = run_one_dataset(
        f"uci_poker_subsample_{args.uci_max}",
        ds_uci,
        alpha_uci,
        sl_uci,
        args.epochs_other_hard,
        args.epochs_other_soft,
    )
    triples.append((block_u, rh_u, rs_u, rl_u, yt_u))

    # 3) Balanced parentheses
    pc = args.parens_csv if os.path.isabs(args.parens_csv) else os.path.join(root, args.parens_csv)
    ds_p, alpha_p, sl_p = load_csv_tokens(pc, max_samples=None)
    block_p, rh_p, rs_p, rl_p, yt_p = run_one_dataset(
        "balanced_parentheses",
        ds_p,
        alpha_p,
        sl_p,
        args.epochs_other_hard,
        args.epochs_other_soft,
    )
    triples.append((block_p, rh_p, rs_p, rl_p, yt_p))

    os.makedirs(args.out_dir, exist_ok=True)
    all_blocks = []
    for b, rh, rs, rl, yt in triples:
        all_blocks.append(b)
        safe = b["dataset"].replace(" ", "_")
        plot_metrics_bars(b, os.path.join(args.out_dir, f"metrics_{safe}.png"))
        plot_roc_three(yt, rh, rs, rl, b["dataset"], os.path.join(args.out_dir, f"roc_{safe}.png"))

    payload = {
        "splits": "70/15/15 train/val/test, seed 42",
        "datasets": all_blocks,
    }
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.json_out} and plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
