#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base, "training", "results_hard_layers.json")
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results_by_layers"]
    layers = [1, 2, 3]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    # 1) Annotated grouped bars
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, layer in enumerate(layers):
        vals = [results[str(layer)][m] for m in metrics]
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=f"L={layer}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Simplified Transformer (Hard): 1 vs 2 vs 3 Layers")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_layers_grouped_bars_annotated.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 2) Metric trend across layers
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label in zip(metrics, metric_labels):
        vals = [results[str(layer)][m] for layer in layers]
        ax.plot(layers, vals, marker="o", linewidth=2, label=label)
    ax.set_xticks(layers)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric Trends by Layer Depth")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_metric_trends_by_layer.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # 3) Confusion matrices side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = max(max(max(r["confusion_matrix"])) for r in results.values())
    for ax, layer in zip(axes, layers):
        cm = np.array(results[str(layer)]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"L={layer}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Confusion Matrices by Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_confusion_matrices_by_layer.png"), dpi=180, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

