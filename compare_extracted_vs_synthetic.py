#!/usr/bin/env python3
"""
Compare extracted (unfolded) rules/formula from transformer against synthetic formula.

Workflow:
1) Load synthetic formula + dataset.
2) Load checkpoint OR (optionally) quick-train a compatible SimplifiedTransformer.
3) Extract rules/formula with LTLExtractor (memoized unfolding).
4) Compare extracted vs synthetic:
   - normalized string equality
   - semantic agreement on held-out test traces
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from extractor import LTLExtractor
from transformer import SimplifiedTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_formula(s: str) -> str:
    return "".join(s.split())


def atom_pair_eval(seq: Sequence[str], a: str, b: str, c: str) -> bool:
    """(¬a) U (SeenPair(b,c)) with first-completion semantics used in repo."""
    pos_b = next((i for i, x in enumerate(seq) if x == b), None)
    pos_c = next((i for i, x in enumerate(seq) if x == c), None)
    if pos_b is None or pos_c is None:
        return False
    first_bc = max(pos_b, pos_c)
    return a not in seq[:first_bc]


def atom_single_eval(seq: Sequence[str], a: str, b: str) -> bool:
    """(¬a) U (⊤ U b) with first occurrence of b."""
    pos_b = next((i for i, x in enumerate(seq) if x == b), None)
    if pos_b is None:
        return False
    return a not in seq[:pos_b]


def synthetic_formula_to_string(formula: List[List[List[str]]]) -> str:
    """
    Generic serializer:
      OR over clauses, each clause is AND over atoms.
      Atom arity 3 => pair witness
      Atom arity 2 => single witness
    """
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
        if not conj:
            continue
        disj.append(" ∧ ".join(conj) if len(conj) > 1 else conj[0])
    return " ∨ ".join(disj) if disj else "⊥"


def eval_synthetic_formula(seq: Sequence[str], formula: List[List[List[str]]]) -> bool:
    """
    OR over clauses, AND within clause.
    """
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


def load_dataset_csv(path: Path) -> Tuple[List[List[str]], List[int]]:
    df = pd.read_csv(path)
    seqs = [str(s).split(",") for s in df["sequence"].tolist()]
    labels = [int(x) for x in df["label"].tolist()]
    return seqs, labels


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_split(n: int, seed: int) -> Split:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return Split(
        train_idx=idx[:n_train],
        val_idx=idx[n_train:n_train + n_val],
        test_idx=idx[n_train + n_val:],
    )


class SeqDataset(Dataset):
    def __init__(self, seqs: List[List[str]], labels: List[int], stoi: Dict[str, int], omega_idx: int):
        self.x = []
        self.y = labels
        for s in seqs:
            ids = [stoi[t] for t in s] + [omega_idx]
            self.x.append(ids)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)


def infer_model_from_state_dict(sd: Dict[str, torch.Tensor], threshold: float = 0.0) -> SimplifiedTransformer:
    vocab_size, d_model = sd["W_enc"].shape
    q_keys = sorted([k for k in sd.keys() if k.startswith("Q_layers.") and k.endswith(".weight")])
    num_layers = len(q_keys)
    d_prime = sd[q_keys[0]].shape[0]
    model = SimplifiedTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_prime=d_prime,
        num_layers=num_layers,
        threshold=threshold,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def quick_train_model(
    seqs: List[List[str]],
    labels: List[int],
    alphabet: List[str],
    seed: int,
    epochs: int,
    d_model: int = 32,
    d_prime: int = 16,
    num_layers: int = 1,
    lr: float = 1e-3,
) -> Tuple[SimplifiedTransformer, Split]:
    set_seed(seed)
    split = make_split(len(seqs), seed)
    omega_idx = len(alphabet)
    stoi = {s: i for i, s in enumerate(alphabet)}

    train_ds = SeqDataset([seqs[i] for i in split.train_idx], [labels[i] for i in split.train_idx], stoi, omega_idx)
    val_ds = SeqDataset([seqs[i] for i in split.val_idx], [labels[i] for i in split.val_idx], stoi, omega_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = SimplifiedTransformer(
        vocab_size=len(alphabet) + 1,
        d_model=d_model,
        d_prime=d_prime,
        num_layers=num_layers,
        threshold=0.0,
    )
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            logits = model.forward_raw(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                pred = (model.forward_raw(x) > model.threshold).long().cpu().numpy().tolist()
                all_p.extend(pred)
                all_y.extend(y.long().cpu().numpy().tolist())
        f1 = f1_score(all_y, all_p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    return model, split


def main() -> None:
    p = argparse.ArgumentParser(description="Compare extracted vs synthetic formula")
    p.add_argument("--formula_json", required=True)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfold_budget", type=int, default=50000)
    p.add_argument("--quick_train_if_needed", action="store_true")
    p.add_argument("--quick_train_epochs", type=int, default=8)
    p.add_argument("--quick_train_layers", type=int, default=1)
    p.add_argument("--quick_train_d_model", type=int, default=32)
    p.add_argument("--quick_train_d_prime", type=int, default=16)
    p.add_argument("--output_json", default="comparison_extracted_vs_synthetic.json")
    args = p.parse_args()

    formula_path = Path(args.formula_json)
    dataset_path = Path(args.dataset_csv)

    with open(formula_path, "r", encoding="utf-8") as f:
        formula_payload = json.load(f)
    formula = formula_payload["formula"]
    alphabet = formula_payload["metadata"]["alphabet"]

    seqs, labels = load_dataset_csv(dataset_path)
    split = make_split(len(seqs), args.seed)

    model = None
    model_source = ""
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        if "W_enc" not in sd:
            raise ValueError("Checkpoint doesn't look like SimplifiedTransformer state_dict.")
        ck_vocab = sd["W_enc"].shape[0]
        needed_vocab = len(alphabet) + 1
        if ck_vocab == needed_vocab:
            model = infer_model_from_state_dict(sd, threshold=0.0)
            model_source = f"checkpoint:{args.checkpoint}"
        elif args.quick_train_if_needed:
            print(f"[info] Checkpoint vocab={ck_vocab} incompatible with dataset vocab={needed_vocab}; quick-training.")
        else:
            raise ValueError(
                f"Checkpoint vocab ({ck_vocab}) incompatible with formula alphabet ({needed_vocab-1}). "
                "Use --quick_train_if_needed."
            )

    if model is None:
        model, split = quick_train_model(
            seqs, labels, alphabet, args.seed,
            epochs=args.quick_train_epochs,
            d_model=args.quick_train_d_model,
            d_prime=args.quick_train_d_prime,
            num_layers=args.quick_train_layers,
        )
        model_source = (
            f"quick_train:{args.quick_train_epochs}epochs,"
            f"L={args.quick_train_layers},d={args.quick_train_d_model},d'={args.quick_train_d_prime}"
        )

    extractor = LTLExtractor(
        model=model,
        alphabet=alphabet,
        threshold=model.threshold,
        unfold_node_budget=args.unfold_budget,
    )
    extracted = extractor.extract_formula()
    extracted_formula = extracted["formula"]
    extracted_ast = extracted.get("formula_ast")

    synthetic_formula = synthetic_formula_to_string(formula)

    # Semantic agreement on test split
    test_idx = split.test_idx
    n = len(test_idx)
    agree = 0
    syn_acc = 0
    ext_acc = 0
    for i in test_idx:
        s = seqs[i]
        y = labels[i]
        y_syn = int(eval_synthetic_formula(s, formula))
        y_ext = int(eval_ast_formula(s, extracted_ast)) if extracted_ast is not None else 0
        agree += int(y_syn == y_ext)
        syn_acc += int(y_syn == y)
        ext_acc += int(y_ext == y)

    report = {
        "model_source": model_source,
        "formula_json": str(formula_path),
        "dataset_csv": str(dataset_path),
        "synthetic_formula": synthetic_formula,
        "extracted_formula": extracted_formula,
        "normalized_exact_match": normalize_formula(synthetic_formula) == normalize_formula(extracted_formula),
        "test_semantic_agreement": agree / max(1, n),
        "test_synthetic_formula_accuracy_against_labels": syn_acc / max(1, n),
        "test_extracted_formula_accuracy_against_labels": ext_acc / max(1, n),
        "n_test": n,
        "unfold_stats": extracted.get("unfold_stats", {}),
    }

    out = Path(args.output_json)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()

