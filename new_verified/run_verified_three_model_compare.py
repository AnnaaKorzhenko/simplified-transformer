#!/usr/bin/env python3
"""
Train hard-attention simplified transformer, softmax transformer, and logistic regression
on new_verified/data (same 70/15/15 split, seed 42). Writes JSON + plots under new_verified/.
Run from project root: python3 new_verified/run_verified_three_model_compare.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from compare_three_models import run_one_dataset  # noqa: E402
from training.utils import load_single_dataset  # noqa: E402


def plot_metrics_bars_annotated(block: dict, out_path: str) -> None:
    models = ["hard", "softmax", "logistic_regression"]
    labels = ["Hard-attn\n(simplified)", "Softmax\n(transformer)", "Logistic\nregression"]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_xtick = ["Acc.", "Prec.", "Rec.", "F1", "AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(models):
        vals = [block["models"][m][k] for k in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=labels[i])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_xtick)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(block["dataset"])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_roc_three(y_test, r_hard, r_soft, r_lr, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    curves = [
        ("Hard-attn (simplified)", r_hard["y_pred_proba"], r_hard["auc"], "#2ca02c"),
        ("Softmax transformer", r_soft["y_pred_proba"], r_soft["auc"], "#d62728"),
        ("Logistic regression", r_lr["y_pred_proba"], r_lr["auc"], "#1f77b4"),
    ]
    for name, probs, auc, color in curves:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC={float(auc):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    data_dir = os.path.join(ROOT, "new_verified", "data")
    out_train = os.path.join(ROOT, "new_verified", "training")
    out_plots = os.path.join(ROOT, "new_verified", "plots")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    _f, dataset, alphabet, seq_len = load_single_dataset(data_dir, formula_id=1)
    block, rh, rs, rl, yt = run_one_dataset(
        "new_verified_diamond_star (alphabet≤6, BLACK-verified)",
        dataset,
        alphabet,
        seq_len,
        epochs_hard=100,
        epochs_soft=100,
        seed=42,
        num_layers_hard=3,
    )

    payload = {
        "splits": "70/15/15 train/val/test, seed 42",
        "data_dir": "new_verified/data",
        "formula_id": 1,
        "epochs_hard": 100,
        "epochs_soft": 100,
        "num_layers_hard": 3,
        "dataset": block,
    }
    json_path = os.path.join(out_train, "three_model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {json_path}")

    plot_metrics_bars_annotated(
        block, os.path.join(out_plots, "04_three_models_metrics_annotated.png")
    )
    plot_roc_three(
        yt,
        rh,
        rs,
        rl,
        "Verified diamond-star: hard vs softmax vs logistic regression",
        os.path.join(out_plots, "05_three_models_roc.png"),
    )
    print(f"Wrote plots under {out_plots}/")


if __name__ == "__main__":
    main()
