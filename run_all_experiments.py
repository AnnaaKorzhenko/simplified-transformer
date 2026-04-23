#!/usr/bin/env python3
"""
Full experiment suite: pair-witness vs single-witness formulas, simple (m=2) vs
complex (m=3), hard-attention depth sweep, hard vs softmax. NO logistic regression.

Architecture matches LLMsREs: leftmost hard-max attention, single-head, no
positional encoding, no masking, Xℓ = σ(Aℓ·Xℓ−1·Vℓ·Oℓ) + Xℓ−1, readout X[0,0].

Usage:
  python3 run_all_experiments.py --epochs 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def py():
    return sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    r = subprocess.run(cmd, check=False, env=env)
    if r.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(r.returncode)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_datasets(seed: int) -> None:
    configs = [
        ("generated_pair_m2", "generate_diamond_star_dataset.py", "2"),
        ("generated_pair_m3", "generate_diamond_star_dataset.py", "3"),
        ("generated_single_m2", "generate_until_single_dataset.py", "2"),
        ("generated_single_m3", "generate_until_single_dataset.py", "3"),
    ]
    for outdir, script, m in configs:
        run([py(), script,
             "--output_dir", outdir,
             "--formula_id", "1",
             "--alphabet_size", "6",
             "--num_disjunctions", m,
             "--sequence_length", "10",
             "--num_positive", "500",
             "--num_negative", "500",
             "--seed", str(seed)])
    run([py(), "generate_rule5_dataset.py",
         "--output_dir", "generated_rule5_llmsres",
         "--formula_id", "1",
         "--alphabet_size", "6",
         "--sequence_length", "10",
         "--num_positive", "500",
         "--num_negative", "500",
         "--seed", str(seed)])


# ---------------------------------------------------------------------------
# Training helpers (import here so matplotlib is already Agg)
# ---------------------------------------------------------------------------
from training.utils import load_single_dataset, split_dataset_three_way  # noqa: E402
from training.train_transformer_hard import train_transformer_hard  # noqa: E402
from training.train_transformer_softmax import train_transformer_soft  # noqa: E402
from training.significance import mcnemar_chi2_paired  # noqa: E402


def _pack(r: dict) -> dict:
    return {
        "accuracy": float(r["accuracy"]),
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "f1": float(r["f1"]),
        "auc": float(r["auc"]),
        "confusion_matrix": r["confusion_matrix"],
    }


def train_setting(dataset_dir: str, formula_id: int, epochs: int, seed: int,
                  layers_list: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Train hard-attention at multiple depths + softmax, return metrics dict."""
    _, dataset, alphabet, seq_len = load_single_dataset(dataset_dir, formula_id)
    X_tr, y_tr, X_v, y_v, X_te, y_te = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=seed)

    hard_by_layer: dict[str, dict] = {}
    for L in layers_list:
        print(f"\n=== {dataset_dir} hard L={L} ===")
        r = train_transformer_hard(X_tr, y_tr, X_te, y_te, alphabet, seq_len,
                                   epochs=epochs, random_state=seed,
                                   num_layers=L, X_val=X_v, y_val=y_v)
        hard_by_layer[str(L)] = {**_pack(r), "y_pred": r["y_pred"],
                                  "y_pred_proba": r.get("y_pred_proba", [])}

    print(f"\n=== {dataset_dir} softmax ===")
    r_soft = train_transformer_soft(X_tr, y_tr, X_te, y_te, alphabet, seq_len,
                                    epochs=epochs, learning_rate=0.001,
                                    random_state=seed, X_val=X_v, y_val=y_v)

    y_list = list(y_te)

    # McNemar across depths
    layer_keys = sorted(hard_by_layer.keys(), key=int)
    sig_layers: dict = {}
    for i, a in enumerate(layer_keys):
        for b in layer_keys[i+1:]:
            sig_layers[f"L{a}_vs_L{b}"] = mcnemar_chi2_paired(
                y_list, hard_by_layer[a]["y_pred"], hard_by_layer[b]["y_pred"])

    # McNemar hard-best vs softmax
    best_L = max(layer_keys, key=lambda k: hard_by_layer[k]["accuracy"])
    sig_hard_soft = mcnemar_chi2_paired(
        y_list, hard_by_layer[best_L]["y_pred"], r_soft["y_pred"])

    return {
        "dataset_dir": dataset_dir,
        "formula_id": formula_id,
        "alphabet_size": len(alphabet),
        "sequence_length": seq_len,
        "n_test": len(y_list),
        "y_test": y_list,
        "hard_by_layer": hard_by_layer,
        "softmax": {**_pack(r_soft), "y_pred": r_soft["y_pred"],
                     "y_pred_proba": r_soft.get("y_pred_proba", [])},
        "mcnemar_layers": {
            "method": "McNemar chi-square (Yates), asymptotic p-value, same test rows",
            "pairs": sig_layers,
        },
        "mcnemar_hard_vs_softmax": {
            "best_hard_L": int(best_L),
            **sig_hard_soft,
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_layers_bars(setting: dict, title: str, path: str) -> None:
    results = setting["hard_by_layer"]
    layers = sorted(results.keys(), key=int)
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Acc.", "Prec.", "Rec.", "F1", "AUC"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, L in enumerate(layers):
        vals = [results[L][m] for m in metrics]
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=f"L={L}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10); ax.set_ylabel("Score"); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def plot_hard_vs_soft_bars(setting: dict, title: str, path: str) -> None:
    best_L = str(max(setting["hard_by_layer"].keys(), key=lambda k: setting["hard_by_layer"][k]["accuracy"]))
    hard = setting["hard_by_layer"][best_L]
    soft = setting["softmax"]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Acc.", "Prec.", "Rec.", "F1", "AUC"]
    x = np.arange(len(metrics)); width = 0.32
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, d, col) in enumerate([
        (f"Hard L={best_L}", hard, "#2ca02c"), ("Softmax", soft, "#d62728")]):
        vals = [d[m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10); ax.set_ylabel("Score"); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def plot_roc_hard_vs_soft(setting: dict, title: str, path: str) -> None:
    from sklearn.metrics import roc_curve
    best_L = str(max(setting["hard_by_layer"].keys(), key=lambda k: setting["hard_by_layer"][k]["accuracy"]))
    hard = setting["hard_by_layer"][best_L]
    soft = setting["softmax"]
    y = setting["y_test"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d, col in [
        (f"Hard L={best_L}", hard, "#2ca02c"), ("Softmax", soft, "#d62728")]:
        proba = d.get("y_pred_proba", [])
        if proba:
            fpr, tpr, _ = roc_curve(y, proba)
            ax.plot(fpr, tpr, lw=2, color=col, label=f"{name} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def plot_pair_vs_single_comparison(pair: dict, single: dict, title: str, path: str) -> None:
    """Side-by-side bar chart comparing pair and single witness at best L."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Acc.", "Prec.", "Rec.", "F1", "AUC"]
    best_Lp = str(max(pair["hard_by_layer"].keys(), key=lambda k: pair["hard_by_layer"][k]["accuracy"]))
    best_Ls = str(max(single["hard_by_layer"].keys(), key=lambda k: single["hard_by_layer"][k]["accuracy"]))
    dp = pair["hard_by_layer"][best_Lp]
    ds = single["hard_by_layer"][best_Ls]
    x = np.arange(len(metrics)); width = 0.32
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, d, col) in enumerate([
        (f"Pair (L={best_Lp})", dp, "#1f77b4"),
        (f"Single (L={best_Ls})", ds, "#ff7f0e")]):
        vals = [d[m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10); ax.set_ylabel("Score"); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="experiment_results")
    p.add_argument("--skip_generate", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not args.skip_generate:
        print("\n===== Generating datasets =====\n")
        generate_datasets(args.seed)

    configs = [
        ("generated_pair_m2",   "Pair witness, m=2 disjuncts"),
        ("generated_pair_m3",   "Pair witness, m=3 disjuncts"),
        ("generated_single_m2", "Single witness, m=2 disjuncts"),
        ("generated_single_m3", "Single witness, m=3 disjuncts"),
        ("generated_rule5_llmsres", "LLMsREs rule (5) style (head + ♢⋆ until + never)"),
    ]

    all_results: dict[str, dict] = {}
    for ds_dir, desc in configs:
        print(f"\n{'='*60}\n{desc}: {ds_dir}\n{'='*60}")
        res = train_setting(ds_dir, formula_id=1, epochs=args.epochs, seed=args.seed)
        all_results[ds_dir] = res

        tag = ds_dir.replace("generated_", "")
        plot_layers_bars(res, f"Hard-attn depth: {desc}", os.path.join(plots_dir, f"layers_{tag}.png"))
        plot_hard_vs_soft_bars(res, f"Hard vs Softmax: {desc}", os.path.join(plots_dir, f"hard_vs_soft_{tag}.png"))
        plot_roc_hard_vs_soft(res, f"ROC: {desc}", os.path.join(plots_dir, f"roc_{tag}.png"))

    # Pair vs single comparison plots
    for suffix, m_label in [("m2", "m=2"), ("m3", "m=3")]:
        pair_k = f"generated_pair_{suffix}"
        single_k = f"generated_single_{suffix}"
        plot_pair_vs_single_comparison(
            all_results[pair_k], all_results[single_k],
            f"Pair vs Single witness ({m_label})",
            os.path.join(plots_dir, f"pair_vs_single_{suffix}.png"))

    # Strip y_pred/y_test/y_pred_proba for JSON (too large)
    json_safe = {}
    for k, v in all_results.items():
        v2 = dict(v)
        v2.pop("y_test", None)
        hbl = {}
        for lk, lv in v2["hard_by_layer"].items():
            lv2 = dict(lv); lv2.pop("y_pred", None); lv2.pop("y_pred_proba", None)
            hbl[lk] = lv2
        v2["hard_by_layer"] = hbl
        sm = dict(v2["softmax"]); sm.pop("y_pred", None); sm.pop("y_pred_proba", None)
        v2["softmax"] = sm
        json_safe[k] = v2

    payload = {
        "protocol": "70/15/15 train/val/test",
        "epochs": args.epochs,
        "seed": args.seed,
        "architecture": "LLMsREs: leftmost hard-max, single-head, no positional enc, no masking, ReLU, readout X[0,0]",
        "models": "hard-attention (L=1,2,3) and softmax-attention (L=3). NO logistic regression.",
        "settings": json_safe,
    }
    json_path = os.path.join(args.out_dir, "all_experiments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")
    print(f"Plots in {plots_dir}/")
    print("All experiments finished.")


if __name__ == "__main__":
    main()
