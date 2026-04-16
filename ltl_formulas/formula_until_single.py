"""
Finite-trace formulas: disjunction of (¬a U b) atoms (single-symbol witness).

Contrasts with formula_generator.py, which uses (¬a U SeenPair(b,c)) with a
derived two-symbol witness. Here the right-hand side is the LLMsREs §3 primitive
``(⊤ U b)'' (singleton ``eventually b''), matching the paper's surface syntax.
"""

from __future__ import annotations

import csv
import json
import random
from typing import Dict, List, Tuple

# Internal format: list of disjuncts; each disjunct is [[a, b]] (one pair).


class FormulaGeneratorUntilSingle:
    """Samples disjunction of (¬a U (⊤ U b)) with distinct a, b per disjunct."""

    def __init__(
        self,
        alphabet_size: int,
        num_disjunctions: int,
        num_conjunctions: int = 1,
        alphabet: List[str] | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)
        self.alphabet_size = alphabet_size
        self.num_disjunctions = num_disjunctions
        self.num_conjunctions = 1
        if alphabet is not None:
            if len(alphabet) < alphabet_size:
                raise ValueError("alphabet shorter than alphabet_size")
            self.alphabet = alphabet[:alphabet_size]
        else:
            self.alphabet = [f"p{i+1}" for i in range(alphabet_size)]

    def generate_formula(self) -> List[List[List[str]]]:
        formula: List[List[List[str]]] = []
        for _ in range(self.num_disjunctions):
            a, b = random.sample(self.alphabet, 2)
            formula.append([[a, b]])
        return formula

    def formula_to_string(self, formula: List[List[List[str]]]) -> str:
        parts = []
        for clause in formula:
            if not clause:
                continue
            a, b = clause[0][0], clause[0][1]
            parts.append(f"(¬{a} U (⊤ U {b}))")
        return " ∨ ".join(parts)


class FormulaEvaluatorUntilSingle:
    """(¬a U b): first b at index j; no a in trace[:j]."""

    @staticmethod
    def evaluate_atom(sequence: List[str], a: str, b: str) -> bool:
        if not sequence:
            return False
        pos_b = None
        for i, sym in enumerate(sequence):
            if sym == b and pos_b is None:
                pos_b = i
                break
        if pos_b is None:
            return False
        return a not in sequence[:pos_b]

    @classmethod
    def evaluate_formula(cls, sequence: List[str], formula: List[List[List[str]]]) -> bool:
        for clause in formula:
            if not clause:
                continue
            a, b = clause[0][0], clause[0][1]
            if cls.evaluate_atom(sequence, a, b):
                return True
        return False


class SyntheticDataGeneratorUntilSingle:
    def __init__(self, alphabet: List[str], seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.alphabet = alphabet

    def generate_random_sequence(self, length: int) -> List[str]:
        return random.choices(self.alphabet, k=length)

    def generate_sequence_satisfying_formula(
        self,
        formula: List[List[List[str]]],
        length: int,
        max_attempts: int = 10000,
    ) -> List[str]:
        target = random.choice(formula)
        a, b = target[0][0], target[0][1]
        seq = [random.choice(self.alphabet) for _ in range(length)]
        max_pos = max(1, (2 * length) // 3)
        pos_b = random.randint(0, max_pos - 1) if length > 1 else 0
        seq[pos_b] = b
        for i in range(pos_b):
            if seq[i] == a:
                seq[i] = random.choice([x for x in self.alphabet if x != a])
        if FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
            return seq
        for _ in range(max_attempts):
            seq = self.generate_random_sequence(length)
            if FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                return seq
        return seq

    def generate_sequence_not_satisfying_formula(
        self,
        formula: List[List[List[str]]],
        length: int,
        max_attempts: int = 10000,
    ) -> List[str]:
        for _ in range(min(1000, max_attempts)):
            seq = self.generate_random_sequence(length)
            if not FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                return seq
        seq = [random.choice(self.alphabet) for _ in range(length)]
        for clause in formula:
            a, b = clause[0][0], clause[0][1]
            pos_a = random.randint(0, max(0, length // 3)) if length > 1 else 0
            seq[pos_a] = a
            if pos_a + 1 < length:
                pos_b = random.randint(pos_a + 1, length - 1)
                seq[pos_b] = b
        if not FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
            return seq
        for clause in formula:
            a, b = clause[0][0], clause[0][1]
            seq = [random.choice(self.alphabet) for _ in range(length)]
            seq[0] = a
            if length > 1:
                seq[-1] = b
        return seq

    def generate_dataset(
        self,
        formula: List[List[List[str]]],
        sequence_length: int,
        num_positive: int,
        num_negative: int,
        seed: int | None = None,
    ) -> List[Tuple[List[str], int]]:
        if seed is not None:
            random.seed(seed)
        out: List[Tuple[List[str], int]] = []
        for _ in range(num_positive):
            seq = self.generate_sequence_satisfying_formula(formula, sequence_length)
            if FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                out.append((seq, 1))
            else:
                for _ in range(10):
                    seq = self.generate_sequence_satisfying_formula(formula, sequence_length)
                    if FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                        out.append((seq, 1))
                        break
                else:
                    out.append((seq, 0))
        for _ in range(num_negative):
            seq = self.generate_sequence_not_satisfying_formula(formula, sequence_length)
            if not FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                out.append((seq, 0))
            else:
                for _ in range(10):
                    seq = self.generate_sequence_not_satisfying_formula(formula, sequence_length)
                    if not FormulaEvaluatorUntilSingle.evaluate_formula(seq, formula):
                        out.append((seq, 0))
                        break
                else:
                    out.append((seq, 1))
        return out


def save_formula_json_until_single(
    formula: List[List[List[str]]], filename: str, metadata: Dict | None = None
) -> None:
    data = {"formula": formula, "metadata": metadata or {}}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_formula_json_until_single(filename: str) -> Tuple[List[List[List[str]]], dict]:
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data["formula"], data.get("metadata", {})


def save_dataset_csv(dataset: List[Tuple[List[str], int]], filename: str, delimiter: str = ",") -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "label"])
        for seq, label in dataset:
            w.writerow([delimiter.join(seq), label])
