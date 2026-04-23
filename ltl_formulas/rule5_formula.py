"""
LLMsREs-style extraction rule (paper eq. (5) shape), for synthetic data.

Rule template (readout / begin anchoring as in extractor.py docstring):
  a ∧ ♢⋆(⋀_{b'∈β_{a,j}} (¬b' U b)) ∧ ♢⋆(⋀_{b'∈β_{a,j'}, j'>j} ¬(⊤ U b'))  →  s_{a,b}

With ♢⋆φ evaluated at the start of the trace (position 0), and global label:
sequence is positive iff the antecedent holds at position 0.

Finite-trace semantics used here:
- (¬b' U b): first occurrence of b appears at index j, and no b' occurs in seq[0:j].
- ¬(⊤ U b'): symbol b' never appears anywhere on the trace.
- Conjunction over β_{a,j}: all (¬b' U b) must hold; equivalently, if j is the index of
  the first b, then no b' ∈ β_{a,j} appears in seq[0:j].
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


def evaluate_rule5_instance(sequence: List[str], rule: Dict[str, Any]) -> bool:
    """
    Args:
        sequence: list of symbol strings
        rule: dict with keys:
            head (str): symbol a at position 0
            witness (str): symbol b
            beta_same_partition: list of b' for the first ♢⋆ conjunct
            never_symbols: list of b' for ¬(⊤ U b') (must never appear)
    """
    if not sequence:
        return False
    head = rule["head"]
    witness = rule["witness"]
    beta = list(rule.get("beta_same_partition", []))
    never_syms = list(rule.get("never_symbols", []))

    if sequence[0] != head:
        return False
    for s in never_syms:
        if s in sequence:
            return False

    pos_b = next((i for i, x in enumerate(sequence) if x == witness), None)
    if pos_b is None:
        return False
    prefix = sequence[:pos_b]
    for bprime in beta:
        if bprime in prefix:
            return False
    return True


def rule5_to_paper_string(rule: Dict[str, Any]) -> str:
    """Human-readable string matching the paper-style rule."""
    a = rule["head"]
    b = rule["witness"]
    beta = rule.get("beta_same_partition", [])
    never = rule.get("never_symbols", [])
    if beta:
        first = " ∧ ".join(f"(¬{bp} U {b})" for bp in beta)
        inner1 = f"♢⋆({first})"
    else:
        inner1 = f"♢⋆(⊤ U {b})"
    if never:
        second = " ∧ ".join(f"¬(⊤ U {bp})" for bp in never)
        inner2 = f"♢⋆({second})"
    else:
        inner2 = "♢⋆(⊤)"
    return f"{a} ∧ {inner1} ∧ {inner2} → s_{{{a},{b}}}"


def save_rule5_formula_json(
    path: str,
    rule: Dict[str, Any],
    alphabet: List[str],
    sequence_length: int,
    description: str = "",
) -> None:
    import json

    metadata: Dict[str, Any] = {
        "alphabet": alphabet,
        "sequence_length": sequence_length,
        "formula_kind": "rule5_llmsres",
        "rule5": rule,
        "rule5_string": rule5_to_paper_string(rule),
    }
    if description:
        metadata["description"] = description
    payload = {"formula": [], "metadata": metadata}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class Rule5SyntheticGenerator:
    """Generate labeled sequences for one fixed rule5 instance."""

    def __init__(self, alphabet: List[str], rule: Dict[str, Any], seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.alphabet = alphabet
        self.rule = rule
        self.head = rule["head"]
        self.witness = rule["witness"]
        self.beta = list(rule.get("beta_same_partition", []))
        self.never = list(rule.get("never_symbols", []))

    def _random_fill(self, length: int) -> List[str]:
        return [random.choice(self.alphabet) for _ in range(length)]

    def generate_positive(self, length: int) -> List[str]:
        seq = self._random_fill(length)
        seq[0] = self.head
        # Place first witness after any forbidden-in-prefix symbols are avoided
        max_pos = max(1, length - 1)
        j = random.randint(1, max_pos - 1) if length > 2 else 1
        # Ensure beta symbols not in prefix [0:j)
        for i in range(j):
            if seq[i] in self.beta:
                seq[i] = random.choice([x for x in self.alphabet if x not in self.beta])
        seq[j] = self.witness
        # Remove never_symbols from entire sequence
        for i in range(length):
            if seq[i] in self.never:
                seq[i] = random.choice([x for x in self.alphabet if x not in self.never])
        # Head might have been overwritten only at i>0
        seq[0] = self.head
        if not evaluate_rule5_instance(seq, self.rule):
            return self.generate_positive(length)
        return seq

    def generate_negative(self, length: int) -> List[str]:
        modes = [0, 1, 3]
        if self.beta:
            modes.append(2)
        mode = random.choice(modes)
        seq = self._random_fill(length)
        if mode == 0:
            seq[0] = random.choice([x for x in self.alphabet if x != self.head])
        elif mode == 1:
            seq[0] = self.head
            if self.never:
                s = random.choice(self.never)
                seq[random.randint(0, length - 1)] = s
        elif mode == 2:
            seq[0] = self.head
            for s in self.never:
                while s in seq:
                    idx = seq.index(s)
                    seq[idx] = random.choice([x for x in self.alphabet if x != s])
            j = random.randint(1, length - 1)
            seq[j] = self.witness
            bp = random.choice(self.beta)
            k = random.randint(0, j - 1) if j > 0 else 0
            seq[k] = bp
        else:
            seq[0] = self.head
            for s in self.never:
                while s in seq:
                    idx = seq.index(s)
                    seq[idx] = random.choice([x for x in self.alphabet if x != s])
            seq = [x if x != self.witness else random.choice(self.alphabet) for x in seq]
            seq[0] = self.head
        if evaluate_rule5_instance(seq, self.rule):
            return self.generate_negative(length)
        return seq

    def generate_dataset(
        self,
        sequence_length: int,
        num_positive: int,
        num_negative: int,
    ) -> List[Tuple[List[str], int]]:
        out: List[Tuple[List[str], int]] = []
        for _ in range(num_positive):
            seq = self.generate_positive(sequence_length)
            out.append((seq, 1))
        for _ in range(num_negative):
            seq = self.generate_negative(sequence_length)
            out.append((seq, 0))
        return out
