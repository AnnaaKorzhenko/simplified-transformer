#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from extractor import LTLExtractor
from transformer import SimplifiedTransformer as TorchTransformer
from training.train_transformer_hard import train_transformer_hard
from training.train_transformer_softmax import train_transformer_soft
from training.utils import split_dataset_three_way
from ltl_formulas.formula_generator import sequence_satisfies_payload
from ltl_formulas.rule5_formula import rule5_to_paper_string


def normalize_formula(s: str) -> str:
    return "".join(s.split())


def load_formula_json(path: Path) -> Tuple[List[List[List[str]]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["formula"], payload.get("metadata", {})


def load_dataset_csv(path: Path) -> List[Tuple[List[str], int]]:
    df = pd.read_csv(path)
    return [(str(s).split(","), int(y)) for s, y in zip(df["sequence"], df["label"])]


def atom_pair_eval(seq: Sequence[str], a: str, b: str, c: str) -> bool:
    pos_b = next((i for i, x in enumerate(seq) if x == b), None)
    pos_c = next((i for i, x in enumerate(seq) if x == c), None)
    if pos_b is None or pos_c is None:
        return False
    first_bc = max(pos_b, pos_c)
    return a not in seq[:first_bc]


def atom_single_eval(seq: Sequence[str], a: str, b: str) -> bool:
    pos_b = next((i for i, x in enumerate(seq) if x == b), None)
    if pos_b is None:
        return False
    return a not in seq[:pos_b]


def eval_synthetic_formula(seq: Sequence[str], formula: List[List[List[str]]]) -> bool:
    for clause in formula:
        if not clause:
            continue
        ok = True
        for atom in clause:
            if len(atom) == 3:
                ok = ok and atom_pair_eval(seq, atom[0], atom[1], atom[2])
            elif len(atom) == 2:
                ok = ok and atom_single_eval(seq, atom[0], atom[1])
            else:
                ok = False
        if ok:
            return True
    return False


def synthetic_formula_to_string(formula: List[List[List[str]]]) -> str:
    disj = []
    for clause in formula:
        conj = []
        for atom in clause:
            if len(atom) == 3:
                a, b, c = atom
                conj.append(f"(¬{a} U ((⊤ U {b}) ∧ (⊤ U {c})))")
            elif len(atom) == 2:
                a, b = atom
                conj.append(f"(¬{a} U (⊤ U {b}))")
        if conj:
            disj.append(" ∧ ".join(conj) if len(conj) > 1 else conj[0])
    return " ∨ ".join(disj) if disj else "⊥"


def eval_ast_at(ast: Any, seq: Sequence[str], i: int) -> bool:
    kind = ast[0]
    if kind == "sym":
        return i < len(seq) and seq[i] == ast[1]
    if kind == "top":
        return True
    if kind == "bot":
        return False
    if kind == "not":
        return not eval_ast_at(ast[1], seq, i)
    if kind == "and":
        return all(eval_ast_at(x, seq, i) for x in ast[1])
    if kind == "or":
        return any(eval_ast_at(x, seq, i) for x in ast[1])
    if kind == "diamond_star":
        return eval_ast_at(ast[1], seq, 0)
    if kind == "u":
        left, right = ast[1], ast[2]
        n = len(seq)
        for j in range(i, n):
            if eval_ast_at(right, seq, j):
                if all(eval_ast_at(left, seq, k) for k in range(i, j)):
                    return True
        return False
    return False


def eval_ast_formula(seq: Sequence[str], ast: Any) -> bool:
    return eval_ast_at(ast, seq, 0)


def convert_numpy_hard_to_torch(np_model: Any, threshold: float = 0.0) -> TorchTransformer:
    vocab_size = np_model.W_enc.shape[0] + 1  # add omega row
    d_model = np_model.d
    d_prime = np_model.d_prime
    num_layers = np_model.L
    tm = TorchTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_prime=d_prime,
        num_layers=num_layers,
        threshold=threshold,
    )
    with torch.no_grad():
        w_enc = np_model.W_enc
        omega = np.zeros((1, w_enc.shape[1]), dtype=w_enc.dtype)
        w_full = np.concatenate([w_enc, omega], axis=0)
        tm.W_enc.copy_(torch.from_numpy(w_full).float())
        for i in range(num_layers):
            tm.Q_layers[i].weight.copy_(torch.from_numpy(np_model.Q_layers[i].T).float())
            tm.K_layers[i].weight.copy_(torch.from_numpy(np_model.K_layers[i].T).float())
            tm.V_layers[i].weight.copy_(torch.from_numpy(np_model.V_layers[i].T).float())
            tm.O_layers[i].weight.copy_(torch.from_numpy(np_model.O_layers[i].T).float())
        tm.threshold = float(np_model.threshold)
    tm.eval()
    return tm


def one_setting(
    name: str,
    formula_json: Path,
    dataset_csv: Path,
    epochs_hard: int,
    epochs_soft: int,
    layers_hard: int,
    layers_soft: int,
    unfold_budget: int,
    seed: int,
) -> Dict[str, Any]:
    formula, meta = load_formula_json(formula_json)
    dataset = load_dataset_csv(dataset_csv)
    alphabet = meta["alphabet"]
    seq_len = int(meta.get("sequence_length", max(len(s) for s, _ in dataset)))
    X = [s for s, _ in dataset]
    y = [lbl for _, lbl in dataset]

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_three_way(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=seed
    )

    hard = train_transformer_hard(
        X_train, y_train, X_test, y_test, alphabet, seq_len,
        epochs=epochs_hard, random_state=seed, num_layers=layers_hard, X_val=X_val, y_val=y_val
    )
    soft = train_transformer_soft(
        X_train, y_train, X_test, y_test, alphabet, seq_len,
        epochs=epochs_soft, random_state=seed, num_layers=layers_soft, X_val=X_val, y_val=y_val
    )

    torch_model = convert_numpy_hard_to_torch(hard["model"], threshold=float(hard["model"].threshold))
    extractor = LTLExtractor(
        model=torch_model,
        alphabet=alphabet,
        threshold=torch_model.threshold,
        unfold_node_budget=unfold_budget,
    )
    extracted = extractor.extract_formula()
    extracted_formula = extracted["formula"]
    extracted_ast = extracted["formula_ast"]
    if meta.get("formula_kind") == "rule5_llmsres":
        synthetic_formula = meta.get("rule5_string") or rule5_to_paper_string(
            meta["rule5"]
        )
    else:
        synthetic_formula = synthetic_formula_to_string(formula)

    syn_test = [int(sequence_satisfies_payload(seq, formula, meta)) for seq in X_test]
    ext_test = [int(eval_ast_formula(seq, extracted_ast)) for seq in X_test]
    soft_test = [int(v) for v in soft["y_pred"]]
    hard_test = [int(v) for v in hard["y_pred"]]
    y_test_i = [int(v) for v in y_test]

    agree_ext_syn = float(np.mean([a == b for a, b in zip(ext_test, syn_test)]))
    agree_soft_syn = float(np.mean([a == b for a, b in zip(soft_test, syn_test)]))
    agree_hard_syn = float(np.mean([a == b for a, b in zip(hard_test, syn_test)]))

    return {
        "setting": name,
        "formula_json": str(formula_json),
        "dataset_csv": str(dataset_csv),
        "n_total": len(X),
        "n_test": len(X_test),
        "hard_layers": layers_hard,
        "soft_layers": layers_soft,
        "epochs_hard": epochs_hard,
        "epochs_soft": epochs_soft,
        "hard_model_accuracy": float(hard["accuracy"]),
        "soft_model_accuracy": float(soft["accuracy"]),
        "hard_vs_synthetic_agreement_test": agree_hard_syn,
        "soft_vs_synthetic_agreement_test": agree_soft_syn,
        "extracted_vs_synthetic_agreement_test": agree_ext_syn,
        "extracted_vs_labels_accuracy_test": float(np.mean([a == b for a, b in zip(ext_test, y_test_i)])),
        "normalized_exact_match": normalize_formula(extracted_formula) == normalize_formula(synthetic_formula),
        "unfold_stats": extracted.get("unfold_stats", {}),
        "synthetic_formula": synthetic_formula,
        "extracted_formula": extracted_formula,
    }


def write_markdown_table(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("| setting | hard acc | soft acc | hard~syn | soft~syn | extracted~syn | extracted acc | exact match | truncated |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in rows:
        us = r.get("unfold_stats", {})
        trunc = bool(us.get("truncated", False))
        lines.append(
            f"| {r['setting']} | {r['hard_model_accuracy']:.3f} | {r['soft_model_accuracy']:.3f} | "
            f"{r['hard_vs_synthetic_agreement_test']:.3f} | {r['soft_vs_synthetic_agreement_test']:.3f} | "
            f"{r['extracted_vs_synthetic_agreement_test']:.3f} | {r['extracted_vs_labels_accuracy_test']:.3f} | "
            f"{'yes' if r['normalized_exact_match'] else 'no'} | {'yes' if trunc else 'no'} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run extracted-vs-synthetic suite on all synthetic settings.")
    p.add_argument("--epochs_hard", type=int, default=100)
    p.add_argument("--epochs_soft", type=int, default=100)
    p.add_argument("--layers_hard", type=int, default=3)
    p.add_argument("--layers_soft", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfold_budget", type=int, default=60000)
    p.add_argument("--out_json", default="comparison_extracted_vs_synthetic_suite.json")
    p.add_argument("--out_md", default="comparison_extracted_vs_synthetic_suite.md")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    settings = [
        ("pair_m2", root / "generated_pair_m2" / "formulas" / "formula_1.json", root / "generated_pair_m2" / "datasets" / "dataset_1.csv"),
        ("pair_m3", root / "generated_pair_m3" / "formulas" / "formula_1.json", root / "generated_pair_m3" / "datasets" / "dataset_1.csv"),
        ("single_m2", root / "generated_single_m2" / "formulas" / "formula_1.json", root / "generated_single_m2" / "datasets" / "dataset_1.csv"),
        ("single_m3", root / "generated_single_m3" / "formulas" / "formula_1.json", root / "generated_single_m3" / "datasets" / "dataset_1.csv"),
        ("rule5_llmsres", root / "generated_rule5_llmsres" / "formulas" / "formula_1.json", root / "generated_rule5_llmsres" / "datasets" / "dataset_1.csv"),
    ]

    rows = []
    for name, fp, dp in settings:
        print(f"\n=== {name} ===")
        rows.append(
            one_setting(
                name=name,
                formula_json=fp,
                dataset_csv=dp,
                epochs_hard=args.epochs_hard,
                epochs_soft=args.epochs_soft,
                layers_hard=args.layers_hard,
                layers_soft=args.layers_soft,
                unfold_budget=args.unfold_budget,
                seed=args.seed,
            )
        )

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_markdown_table(rows, out_md)
    print(f"\nSaved {out_json}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()

